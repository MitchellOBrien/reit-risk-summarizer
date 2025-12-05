"""Extract risk factors from 10-K filings."""

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup

from ...exceptions import RiskExtractionError

logger = logging.getLogger(__name__)


class RiskFactorExtractor:
    """Extracts Item 1A Risk Factors section from 10-K HTML documents.
    
    This class handles parsing various 10-K HTML formats to extract the
    Risk Factors section text.
    """

    def __init__(
        self, 
        min_length: int = 10_000,
        warn_threshold: int = 30_000,
        raise_on_short: bool = True
    ):
        """Initialize the extractor.
        
        Args:
            min_length: Minimum acceptable length for extracted text.
                       Defaults to 10,000 chars (typical 10-Ks are 50k-150k).
            warn_threshold: Log a warning if extraction is below this threshold.
                           Defaults to 30,000 chars.
            raise_on_short: If True, raise error when below min_length.
                           If False, only log warning. Defaults to True.
        """
        self.min_length = min_length
        self.warn_threshold = warn_threshold
        self.raise_on_short = raise_on_short

    def extract_risk_factors(self, html: str) -> str:
        """Extract Item 1A Risk Factors section from 10-K HTML.
        
        Args:
            html: Raw HTML content of a 10-K filing.
            
        Returns:
            Extracted risk factors text, cleaned and formatted.
            
        Raises:
            RiskExtractionError: If risk factors section cannot be found or extracted.
        """
        if not html or not html.strip():
            raise RiskExtractionError("Empty HTML content provided")
        
        logger.info("Extracting risk factors from 10-K HTML")
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try multiple strategies to find the risk factors section
        risk_text = (
            self._extract_by_item_header(soup) or
            self._extract_by_table_of_contents(soup) or
            self._extract_by_pattern_matching(html)
        )
        
        if not risk_text:
            raise RiskExtractionError(
                "Could not locate Item 1A Risk Factors section in the document",
                details={"html_length": len(html)}
            )
        
        # Clean and format the extracted text
        cleaned_text = self._clean_text(risk_text)
        
        logger.info(f"Successfully extracted risk factors ({len(cleaned_text)} characters)")
        return cleaned_text

    def _extract_by_item_header(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract risk factors by finding Item 1A header.
        
        Strategy: Look for headings containing "Item 1A" or "Risk Factors",
        then extract text until the next item (Item 1B, Item 2, etc.).
        """
        logger.debug("Trying extraction by Item header")
        
        # Patterns to match Item 1A variations
        patterns = [
            r'item\s*1a',
            r'item\s*1\.a',
        ]
        
        # Search in common heading tags and containers
        for tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'div', 'p', 'span', 'td', 'b', 'strong']:
            for tag in soup.find_all(tag_name):
                # Get text including child elements (handles split text like "ITEM 1A." + "RISK FACTORS")
                tag_text = tag.get_text(separator=' ', strip=True).lower()
                
                # Check if this tag contains Item 1A
                for pattern in patterns:
                    if re.search(pattern, tag_text):
                        # Verify it's actually Item 1A (not 11A or similar)
                        if re.search(r'\bitem\s*1a\b|\bitem\s*1\.a\b', tag_text):
                            # Also check if "risk factors" appears nearby (in same element or adjacent)
                            has_risk_factors = 'risk' in tag_text and 'factor' in tag_text
                            
                            logger.debug(f"Found Item 1A header: {tag_text[:100]}")
                            return self._extract_section_content(tag)
        
        return None

    def _extract_section_content(self, start_tag) -> str:
        """Extract content from start tag until next section marker.
        
        Args:
            start_tag: BeautifulSoup tag where section starts.
            
        Returns:
            Text content of the section.
        """
        content_parts = []
        
        # Patterns that indicate the next section (end of risk factors)
        end_patterns = [
            r'\bitem\s*1b\b',
            r'\bitem\s*1\.b\b',
            r'\bitem\s*2\b',
            r'\bunresolved\s*staff\s*comments\b',
        ]
        
        # Start collecting content after the header
        current = start_tag.find_next_sibling()
        
        while current:
            current_text = current.get_text(separator=' ', strip=True).lower()
            
            # Skip "Table of Contents" links (these appear throughout the document)
            if current_text == 'table of contents' or len(current_text) < 5:
                current = current.find_next_sibling()
                continue
            
            # Check if we've reached the next section
            for pattern in end_patterns:
                if re.search(pattern, current_text):
                    # Check if it's a section header (short text, likely a heading)
                    # Also check if it starts with "item" to avoid false positives
                    if len(current_text) < 200 and current_text.startswith('item'):
                        logger.debug(f"Found end marker: {current_text[:50]}")
                        return '\n'.join(content_parts)
            
            # Add this element's text to our content
            text = current.get_text()
            if text.strip():
                content_parts.append(text)
            
            current = current.find_next_sibling()
        
        # If we didn't find an end marker, return what we have
        return '\n'.join(content_parts)

    def _extract_by_table_of_contents(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract risk factors by finding links in table of contents.
        
        Strategy: Look for TOC links to Item 1A, find the anchor/id,
        then extract content at that location.
        """
        logger.debug("Trying extraction by table of contents")
        
        # Find all links that might be TOC entries
        for link in soup.find_all('a', href=True):
            link_text = link.get_text().lower().strip()
            
            # Check if this is a link to Item 1A
            if re.search(r'item\s*1a|risk\s*factors', link_text):
                href = link['href']
                
                # Extract anchor from href (e.g., #item1a)
                if '#' in href:
                    anchor_id = href.split('#')[1]
                    
                    # Find the element with this id
                    target = soup.find(id=anchor_id) or soup.find(attrs={'name': anchor_id})
                    
                    if target:
                        logger.debug(f"Found Item 1A via TOC anchor: {anchor_id}")
                        return self._extract_section_content(target)
        
        return None

    def _extract_by_pattern_matching(self, html: str) -> Optional[str]:
        """Extract risk factors using regex pattern matching.
        
        Strategy: Use regex to find the section in raw HTML, then parse.
        This is a fallback for unusual document structures.
        """
        logger.debug("Trying extraction by pattern matching")
        
        # Pattern to find Item 1A section
        # Looks for "Item 1A" followed by content until "Item 1B" or "Item 2"
        pattern = r'(?i)item\s*1a\.?\s*(?:risk\s*factors)?.*?(?=item\s*1b|item\s*2(?:\.|[^0-9]))'
        
        match = re.search(pattern, html, re.DOTALL)
        if match:
            section_html = match.group(0)
            
            # Parse the matched HTML to extract text
            soup = BeautifulSoup(section_html, 'html.parser')
            text = soup.get_text()
            
            logger.debug(f"Extracted via pattern matching ({len(text)} chars)")
            return text
        
        return None

    def _clean_text(self, text: str) -> str:
        """Clean and format extracted text.
        
        Args:
            text: Raw extracted text.
            
        Returns:
            Cleaned and formatted text.
            
        Raises:
            RiskExtractionError: If text is too short and raise_on_short=True.
        """
        # Remove the Item 1A header itself if present at the start
        text = re.sub(r'^.*?item\s*1a.*?risk\s*factors.*?\n', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone numbers (page numbers)
        text = re.sub(r'\n\s*Table of Contents\s*\n', '\n', text, flags=re.IGNORECASE)
        
        # Remove excessive spacing
        text = text.strip()
        
        # Check length and provide appropriate feedback
        text_len = len(text)
        
        if text_len < self.min_length:
            error_msg = (
                f"Extracted text is too short ({text_len:,} chars < {self.min_length:,} minimum). "
                f"This may indicate extraction captured table of contents or partial content."
            )
            if self.raise_on_short:
                raise RiskExtractionError(
                    error_msg,
                    details={
                        "length": text_len, 
                        "min_length": self.min_length,
                        "preview": text[:500]
                    }
                )
            else:
                logger.warning(error_msg)
        
        elif text_len < self.warn_threshold:
            logger.warning(
                f"Extracted text is shorter than expected ({text_len:,} chars < {self.warn_threshold:,} typical). "
                f"Verify extraction quality."
            )
        
        return text
