"""Fetch 10-K filings from SEC EDGAR."""

import logging
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup
from requests.exceptions import Timeout, RequestException

from ...config import Settings, get_settings
from ...exceptions import InvalidTickerError, SECFetchError

logger = logging.getLogger(__name__)


class SECFetcher:
    """Fetches 10-K filings from SEC EDGAR.
    
    This class handles fetching the latest 10-K filing HTML for a given ticker symbol.
    It respects SEC rate limits and includes proper error handling.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the SEC fetcher.
        
        Args:
            settings: Application settings. If None, uses get_settings().
        """
        self.settings = settings or get_settings()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.settings.sec_api_user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        })
        self.base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        self.last_request_time: Optional[float] = None
        self.rate_limit_delay = 0.1  # 100ms between requests (10 req/sec)
        self.max_retries = 3  # Number of retries for transient errors
        self.retry_backoff = 2.0  # Exponential backoff multiplier

    def fetch_latest_10k(self, ticker: str) -> str:
        """Fetch the latest 10-K filing HTML for a given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'PLD', 'AMT').
            
        Returns:
            Raw HTML content of the 10-K filing.
            
        Raises:
            InvalidTickerError: If ticker is invalid or not found.
            SECFetchError: If fetch fails for any reason.
        """
        ticker = ticker.upper().strip()
        logger.info(f"Fetching latest 10-K for ticker: {ticker}")

        try:
            # Step 1: Get CIK for ticker
            cik = self._get_cik(ticker)
            logger.debug(f"Found CIK {cik} for ticker {ticker}")

            # Step 2: Find latest 10-K filing URL
            filing_url = self._find_latest_filing_url(cik)
            logger.debug(f"Found filing URL: {filing_url}")

            # Step 3: Fetch the filing HTML
            html = self._fetch_filing_html(filing_url)
            logger.info(f"Successfully fetched 10-K for {ticker} ({len(html)} bytes)")

            return html

        except InvalidTickerError:
            raise
        except SECFetchError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching 10-K for {ticker}: {e}")
            raise SECFetchError(
                f"Failed to fetch 10-K for {ticker}",
                details={"ticker": ticker, "error": str(e)}
            )

    def _get_cik(self, ticker: str) -> str:
        """Get CIK (Central Index Key) for a ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            CIK number as string with leading zeros removed.
            
        Raises:
            InvalidTickerError: If ticker is not found.
            SECFetchError: If the request fails.
        """
        self._respect_rate_limit()
        params = {
            'action': 'getcompany',
            'CIK': ticker,
            'type': '10-K',
            'count': '1',
            'output': 'atom'
        }

        try:
            # Wrap the request in retry logic
            def make_request():
                resp = self.session.get(self.base_url, params=params, timeout=30)
                resp.raise_for_status()
                return resp
            
            response = self._retry_request(make_request)

            # Parse the Atom feed to extract CIK
            soup = BeautifulSoup(response.content, 'xml')
            
            # Check if company was found
            company_info = soup.find('company-info')
            if not company_info:
                # Try alternate parsing - look for cik in the feed
                entries = soup.find_all('entry')
                if not entries:
                    raise InvalidTickerError(
                        f"Ticker '{ticker}' not found in SEC database",
                        details={"ticker": ticker}
                    )
            
            # Extract CIK from filing href or company-info
            # CIK appears in URLs like: /cgi-bin/browse-edgar?action=getcompany&CIK=0001045610
            cik_link = soup.find('link', rel='alternate')
            if cik_link and 'href' in cik_link.attrs:
                href = cik_link['href']
                cik_match = re.search(r'CIK=(\d+)', href)
                if cik_match:
                    cik = cik_match.group(1).lstrip('0') or '0'
                    return cik
            
            # Alternative: extract from feed content
            content = response.text
            cik_match = re.search(r'<cik>(\d+)</cik>', content)
            if not cik_match:
                # Last resort: try finding in any URL in the response
                cik_match = re.search(r'/edgar/data/(\d+)/', content)
            
            if cik_match:
                cik = cik_match.group(1).lstrip('0') or '0'
                return cik

            raise InvalidTickerError(
                f"Could not extract CIK for ticker '{ticker}'",
                details={"ticker": ticker}
            )

        except requests.RequestException as e:
            logger.error(f"Network error while fetching CIK for {ticker}: {e}")
            raise SECFetchError(
                f"Failed to fetch CIK for ticker '{ticker}'",
                details={"ticker": ticker, "error": str(e)}
            )

    def _find_latest_filing_url(self, cik: str) -> str:
        """Find the URL of the latest 10-K filing.
        
        Args:
            cik: Company CIK number.
            
        Returns:
            URL to the 10-K filing HTML.
            
        Raises:
            SECFetchError: If no 10-K filing is found or request fails.
        """
        self._respect_rate_limit()

        params = {
            'action': 'getcompany',
            'CIK': cik,
            'type': '10-K',
            'dateb': '',
            'owner': 'exclude',
            'count': '1',
            'output': 'atom'
        }

        try:
            # Wrap the request in retry logic
            def make_request():
                resp = self.session.get(self.base_url, params=params, timeout=30)
                resp.raise_for_status()
                return resp
            
            response = self._retry_request(make_request)

            soup = BeautifulSoup(response.content, 'xml')
            
            # Find the filing entry
            entry = soup.find('entry')
            if not entry:
                raise SECFetchError(
                    f"No 10-K filing found for CIK {cik}",
                    details={"cik": cik}
                )

            # Extract the filing link
            # Look for filing-href which contains the document link
            filing_href = entry.find('filing-href')
            if filing_href:
                filing_url = filing_href.text.strip()
                # The filing-href gives us the index page, we need the actual HTML document
                # Convert from index page to HTML document
                # Example: https://www.sec.gov/cgi-bin/viewer?action=view&cik=1045610&accession_number=0000950170-24-014539
                # We need to get the actual document URL from the index
                return self._extract_document_url(filing_url)
            
            # Alternative: look for link with type html
            links = entry.find_all('link')
            for link in links:
                if link.get('type') == 'text/html':
                    return link.get('href')
            
            # If still not found, try to extract accession number and construct URL
            accession = entry.find('accession-number')
            if accession:
                acc_num = accession.text.strip().replace('-', '')
                # Construct the filing detail URL
                filing_detail_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=exclude&count=1"
                return self._extract_document_url_from_detail(filing_detail_url)

            raise SECFetchError(
                f"Could not find filing URL for CIK {cik}",
                details={"cik": cik}
            )

        except requests.RequestException as e:
            logger.error(f"Network error while finding filing for CIK {cik}: {e}")
            raise SECFetchError(
                f"Failed to find 10-K filing for CIK {cik}",
                details={"cik": cik, "error": str(e)}
            )

    def _extract_document_url(self, index_url: str) -> str:
        """Extract the actual document URL from an index page URL.
        
        Args:
            index_url: URL to the filing index/viewer page.
            
        Returns:
            URL to the actual HTML document.
        """
        # Handle XBRL viewer URLs - extract the actual document path
        # Format: https://www.sec.gov/ix?doc=/Archives/edgar/data/.../file.htm
        if '/ix?doc=' in index_url:
            # Extract the document path from the query parameter
            doc_path = index_url.split('/ix?doc=')[1]
            return f"https://www.sec.gov{doc_path}"
        
        self._respect_rate_limit()
        
        try:
            # Wrap the request in retry logic
            def make_request():
                resp = self.session.get(index_url, timeout=30)
                resp.raise_for_status()
                return resp
            
            response = self._retry_request(make_request)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for the 10-K HTML document link
            # Usually in a table with class 'tableFile'
            table = soup.find('table', {'class': 'tableFile'})
            if table:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header row
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        # Check if this is the main 10-K document (not exhibits)
                        doc_type = cells[3].text.strip()
                        if doc_type == '10-K':
                            doc_link = cells[2].find('a')
                            if doc_link:
                                href = doc_link.get('href')
                                if href:
                                    full_url = f"https://www.sec.gov{href}"
                                    # Recursively handle if this is also an ix viewer URL
                                    if '/ix?doc=' in full_url:
                                        return self._extract_document_url(full_url)
                                    return full_url
            
            # Alternative: look for any link containing .htm (but not -index.htm)
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                if '.htm' in href and '-index.htm' not in href and 'Archives/edgar/data' in href:
                    full_url = f"https://www.sec.gov{href}" if href.startswith('/') else href
                    # Recursively handle if this is also an ix viewer URL
                    if '/ix?doc=' in full_url:
                        return self._extract_document_url(full_url)
                    return full_url

            raise SECFetchError(
                f"Could not extract document URL from index page",
                details={"index_url": index_url}
            )

        except requests.RequestException as e:
            logger.error(f"Error extracting document URL: {e}")
            raise SECFetchError(
                f"Failed to extract document URL",
                details={"index_url": index_url, "error": str(e)}
            )

    def _extract_document_url_from_detail(self, detail_url: str) -> str:
        """Extract document URL from the company detail page.
        
        Args:
            detail_url: URL to company filing detail page.
            
        Returns:
            URL to the actual HTML document.
        """
        self._respect_rate_limit()

        try:
            # Wrap the request in retry logic
            def make_request():
                resp = self.session.get(detail_url, timeout=30)
                resp.raise_for_status()
                return resp
            
            response = self._retry_request(make_request)

            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the Documents button/link
            doc_button = soup.find('a', id='documentsbutton')
            if doc_button:
                href = doc_button.get('href')
                if href:
                    index_url = f"https://www.sec.gov{href}"
                    return self._extract_document_url(index_url)

            raise SECFetchError(
                f"Could not find documents link",
                details={"detail_url": detail_url}
            )

        except requests.RequestException as e:
            logger.error(f"Error fetching detail page: {e}")
            raise SECFetchError(
                f"Failed to fetch detail page",
                details={"detail_url": detail_url, "error": str(e)}
            )

    def _fetch_filing_html(self, url: str) -> str:
        """Fetch the HTML content from a filing URL.
        
        Args:
            url: URL to the 10-K filing document.
            
        Returns:
            HTML content as string.
            
        Raises:
            SECFetchError: If fetch fails.
        """
        self._respect_rate_limit()

        try:
            # Wrap the request in retry logic
            def make_request():
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                return resp
            
            response = self._retry_request(make_request)

            html = response.text
            
            # Basic validation - check if it looks like a 10-K
            html_lower = html.lower()
            if 'risk factors' not in html_lower and 'item 1a' not in html_lower:
                logger.warning(f"Fetched HTML may not be a valid 10-K (no risk factors section found)")

            return html

        except requests.RequestException as e:
            logger.error(f"Error fetching filing HTML from {url}: {e}")
            raise SECFetchError(
                f"Failed to fetch filing HTML",
                details={"url": url, "error": str(e)}
            )

    def _respect_rate_limit(self) -> None:
        """Ensure we don't exceed SEC rate limit of 10 requests per second."""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        
        self.last_request_time = time.time()

    def _retry_request(self, func, *args, **kwargs):
        """Retry a request with exponential backoff for transient errors.
        
        Args:
            func: Function to retry (should make an HTTP request).
            *args: Positional arguments to pass to func.
            **kwargs: Keyword arguments to pass to func.
            
        Returns:
            Result of func(*args, **kwargs).
            
        Raises:
            Exception from func after max retries exhausted.
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (Timeout, RequestException) as e:
                last_exception = e
                
                # Check if this is a retryable error
                is_timeout = isinstance(e, Timeout)
                is_503 = (isinstance(e, requests.HTTPError) and 
                         e.response is not None and 
                         e.response.status_code == 503)
                
                if not (is_timeout or is_503):
                    # Not a transient error, don't retry
                    raise
                
                if attempt < self.max_retries - 1:
                    # Calculate backoff delay: 2^attempt seconds
                    delay = self.retry_backoff ** attempt
                    logger.warning(
                        f"Transient error on attempt {attempt + 1}/{self.max_retries}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Max retries ({self.max_retries}) exhausted: {e}")
        
        # Re-raise the last exception if all retries failed
        raise last_exception
