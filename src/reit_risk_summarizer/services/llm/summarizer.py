"""LLM-powered risk summarization from extracted 10-K risk factors.

This module provides interfaces and implementations for summarizing REIT risk factors
using Large Language Models (Groq, Hugging Face). The summarizer 
identifies the top 5 most material risks from the extracted Item 1A section.

Key Features:
- Abstract interface for multiple LLM providers (API + local models)
- Hugging Face support for local inference (no API/internet required)
- Structured output with ranked risk list
- Prompt version tracking for evaluation
- Configurable model parameters
- Graceful error handling
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import logging

from groq import Groq

from reit_risk_summarizer.config import get_settings
from reit_risk_summarizer.exceptions import LLMSummarizationError as LLMError
from reit_risk_summarizer.services.llm.prompts import v1_0

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RiskSummary:
    """Structured output from risk summarization.
    
    Attributes:
        risks: List of top 5 risks in ranked order (most to least material)
        ticker: REIT ticker symbol
        company_name: Company name
        model: LLM model used (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        prompt_version: Version of prompt template used (for evaluation tracking)
        raw_response: Full LLM response for debugging/auditing
    """
    risks: List[str]
    ticker: str
    company_name: str
    model: str
    prompt_version: str
    raw_response: Optional[str] = None
    
    def __post_init__(self):
        """Validate risks list."""
        if len(self.risks) != 5:
            raise ValueError(f"Expected exactly 5 risks, got {len(self.risks)}")
        if not all(isinstance(r, str) and r.strip() for r in self.risks):
            raise ValueError("All risks must be non-empty strings")


class RiskSummarizer(ABC):
    """Abstract base class for risk summarization using LLMs."""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        prompt_version: str = "v1.0"
    ):
        """Initialize summarizer.
        
        Args:
            model: LLM model identifier
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens in response
            prompt_version: Version of prompt template to use
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_version = prompt_version
    
    @abstractmethod
    def summarize(
        self,
        risk_text: str,
        ticker: str,
        company_name: str
    ) -> RiskSummary:
        """Summarize risk factors into top 5 most material risks.
        
        Args:
            risk_text: Extracted Item 1A risk factors text
            ticker: REIT ticker symbol
            company_name: Company name
            
        Returns:
            RiskSummary with ranked list of 5 risks
            
        Raises:
            LLMError: If summarization fails
        """
        pass
    
    @abstractmethod
    def _build_prompt(self, risk_text: str, ticker: str, company_name: str) -> str:
        """Build the prompt for the LLM.
        
        Args:
            risk_text: Extracted risk factors
            ticker: REIT ticker
            company_name: Company name
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def _parse_response(self, response: str) -> List[str]:
        """Parse LLM response into list of 5 risks.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of exactly 5 risk descriptions
            
        Raises:
            ValueError: If response cannot be parsed into 5 risks
        """
        pass


class GroqRiskSummarizer(RiskSummarizer):
    """Risk summarizer using Groq API (fast inference for Llama, Qwen, Mixtral)."""
    
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        prompt_version: str = "v1.0",
        api_key: Optional[str] = None
    ):
        """Initialize Groq summarizer.
        
        Args:
            model: Groq model (default: llama-3.3-70b-versatile)
                Options: llama-3.3-70b-versatile, llama-3.1-70b-versatile,
                        qwen2.5-72b-versatile, mixtral-8x7b-32768
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            prompt_version: Prompt template version
            api_key: Groq API key (uses settings.groq_api_key if None)
        """
        super().__init__(model, temperature, max_tokens, prompt_version)
        self.client = Groq(api_key=api_key or settings.groq_api_key)
    
    def summarize(
        self,
        risk_text: str,
        ticker: str,
        company_name: str
    ) -> RiskSummary:
        """Summarize risks using Groq API with two-pass chunking if needed."""
        # Check if we need chunking (roughly 10k tokens = 40k chars)
        if len(risk_text) > 40000:
            logger.info(f"Using two-pass summarization for {ticker} ({len(risk_text):,} chars)")
            return self._summarize_with_chunking(risk_text, ticker, company_name)
        else:
            # Single-pass for smaller documents
            return self._summarize_single_pass(risk_text, ticker, company_name)
    
    def _chunk_text(self, text: str, chunk_size: int = 35000) -> List[str]:
        """
        Split text into chunks at sentence boundaries.
        
        Args:
            text: Text to chunk
            chunk_size: Target size per chunk (will adjust to sentence boundaries)
        
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Target end point
            end = min(start + chunk_size, len(text))
            
            # If not at the end, find sentence boundary
            if end < len(text):
                # Look backwards up to 500 chars for sentence end
                for i in range(end, max(start, end - 500), -1):
                    if text[i] in '.!?' and (i + 1 >= len(text) or text[i + 1] in ' \n\t'):
                        end = i + 1
                        break
            
            chunks.append(text[start:end])
            start = end
        
        return chunks
    
    def _summarize_single_pass(
        self,
        risk_text: str,
        ticker: str,
        company_name: str
    ) -> RiskSummary:
        """Single-pass summarization for documents within token limits."""
        import time
        
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                prompt = self._build_prompt(risk_text, ticker, company_name)
                
                logger.info(
                    f"Calling Groq {self.model} for {ticker} "
                    f"(prompt v{self.prompt_version}, attempt {attempt + 1}/{max_retries})"
                )
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=30.0  # Add explicit timeout
                )
                
                raw_response = response.choices[0].message.content
                
                # Parse response - handle if model returns != 5 risks
                try:
                    risks = self._parse_response(raw_response)
                except ValueError as e:
                    if "got" in str(e) and "Expected 5" in str(e):
                        print(f"⚠️ Model returned wrong number of risks for {ticker}, attempting to fix...")
                        all_parsed = self._parse_response_flexible(raw_response)
                        if len(all_parsed) >= 5:
                            risks = all_parsed[:5]
                            print(f"✅ Fixed: took first 5 of {len(all_parsed)} risks")
                        else:
                            print(f"❌ Only found {len(all_parsed)} risks, cannot proceed")
                            raise
                    else:
                        raise
                
                return RiskSummary(
                    risks=risks,
                    ticker=ticker,
                    company_name=company_name,
                    model=self.model,
                    prompt_version=self.prompt_version,
                    raw_response=raw_response
                )
                
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries} for {ticker}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Groq connection failed after {max_retries} attempts for {ticker}")
                    raise LLMError(f"Groq connection failed after {max_retries} retries: {e}") from e
            except ValueError as e:
                logger.error(f"Failed to parse Groq response for {ticker}: {e}")
                raise LLMError(f"Response parsing failed: {e}") from e
            except Exception as e:
                # Other errors (rate limits, API errors, etc.)
                logger.error(f"Groq API error for {ticker}: {e}")
                raise LLMError(f"Groq summarization failed: {e}") from e
    
    def _summarize_with_chunking(
        self,
        risk_text: str,
        ticker: str,
        company_name: str
    ) -> RiskSummary:
        """
        Two-pass summarization for large documents.
        
        Pass 1: Summarize each chunk → ~5 risks per chunk
        Pass 2: Meta-summarize all risks → top 5 overall
        """
        # Pass 1: Chunk and summarize each
        chunks = self._chunk_text(risk_text, chunk_size=35000)
        logger.info(f"Split {ticker} into {len(chunks)} chunks for processing")
        
        all_risks = []
        chunk_responses = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} for {ticker}")
            
            # Summarize this chunk
            chunk_result = self._summarize_single_pass(chunk, ticker, company_name)
            all_risks.extend(chunk_result.risks)
            chunk_responses.append(f"Chunk {i+1}: {chunk_result.raw_response}")
        
        logger.info(f"Pass 1 complete: Found {len(all_risks)} total risks from {len(chunks)} chunks")
        
        # Pass 2: Meta-summarize to top 5
        meta_prompt = self._build_meta_prompt(all_risks, ticker, company_name)
        
        logger.info(f"Pass 2: Meta-summarizing {len(all_risks)} risks to top 5 for {ticker}")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": meta_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=30.0
        )
        
        final_response = response.choices[0].message.content
        
        # Parse response - if model returns >5, take top 5
        try:
            final_risks = self._parse_response(final_response)
        except ValueError as e:
            # If we got more than 5 risks, just take the first 5 (they should be ranked)
            if "got" in str(e) and "Expected 5" in str(e):
                print(f"⚠️ Model returned >5 risks in meta-summary, taking first 5")
                print(f"Raw response from meta-summarization:\n{final_response}\n")
                all_parsed = self._parse_response_flexible(final_response)
                final_risks = all_parsed[:5]
                print(f"✅ Extracted exactly {len(final_risks)} risks from {len(all_parsed)} total")
            else:
                raise
        
        # Combine all responses for debugging
        combined_raw = f"TWO-PASS SUMMARY\n\n"
        combined_raw += "\n\n".join(chunk_responses)
        combined_raw += f"\n\nFINAL META-SUMMARY:\n{final_response}"
        
        return RiskSummary(
            risks=final_risks,
            ticker=ticker,
            company_name=company_name,
            model=f"{self.model} (2-pass)",
            prompt_version=self.prompt_version,
            raw_response=combined_raw
        )
    
    def _build_meta_prompt(self, risks: List[str], ticker: str, company_name: str) -> str:
        """
        Build prompt for Pass 2: selecting top 5 from all chunk risks.
        
        TODO: The model (Llama 3.3-70B) sometimes returns 6 risks despite explicit
        instructions to return exactly 5. We've added post-processing fallback logic
        to handle this, but should revisit to find a better solution:
        - Try different prompt engineering approaches (JSON mode, XML formatting, etc.)
        - Test if other models (Qwen, Mixtral) follow count instructions better
        - Consider using structured output APIs if/when available on Groq
        - Evaluate if temperature adjustments help with adherence to constraints
        """
        risks_text = "\n\n".join([f"{i+1}. {risk}" for i, risk in enumerate(risks)])
        
        return f"""Below are {len(risks)} risk factors identified from analyzing {company_name} ({ticker})'s 10-K filing in multiple sections.

Your task: Select EXACTLY 5 MOST IMPORTANT and MATERIAL risks from this list. Consider:
- Business impact severity
- Likelihood of occurrence  
- Materiality to investors
- Strategic importance

Identified Risks:
{risks_text}

Select the top 5 most important risks and format EXACTLY as shown:
1. [First risk - 1-2 sentences]
2. [Second risk - 1-2 sentences]
3. [Third risk - 1-2 sentences]
4. [Fourth risk - 1-2 sentences]
5. [Fifth risk - 1-2 sentences]

IMPORTANT: Return EXACTLY 5 risks, no more, no less. Do not include any preamble, explanation, or additional text."""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for Groq."""
        if self.prompt_version == "v1.0":
            return v1_0.SYSTEM_PROMPT
        else:
            # Fallback for unknown versions
            return v1_0.SYSTEM_PROMPT
    
    def _build_prompt(self, risk_text: str, ticker: str, company_name: str) -> str:
        """Build user prompt for Groq."""
        if self.prompt_version == "v1.0":
            return v1_0.build_user_prompt(risk_text, ticker, company_name)
        else:
            # Fallback for unknown versions
            return v1_0.build_user_prompt(risk_text, ticker, company_name)
    
    def _parse_response(self, response: str) -> List[str]:
        """Parse Groq response into 5 risks (same logic as OpenAI)."""
        lines = response.strip().split('\n')
        risks = []
        current_risk = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a numbered item (1., 2., etc.)
            if line[0].isdigit() and (line[1] == '.' or line[1] == ')'):
                # Save previous risk if exists
                if current_risk:
                    risks.append(' '.join(current_risk).strip())
                    current_risk = []
                # Start new risk (remove number prefix)
                current_risk.append(line.split('.', 1)[1].strip() if '.' in line else line.split(')', 1)[1].strip())
            else:
                # Continue current risk
                current_risk.append(line)
        
        # Add last risk
        if current_risk:
            risks.append(' '.join(current_risk).strip())
        
        if len(risks) != 5:
            raise ValueError(
                f"Expected 5 risks, got {len(risks)}. "
                f"Response may not follow expected format."
            )
        
        return risks
    
    def _parse_response_flexible(self, response: str) -> List[str]:
        """Parse Groq response into risks without validating count (for fallback).
        
        This is used when the model returns more than 5 risks despite instructions.
        """
        lines = response.strip().split('\n')
        risks = []
        current_risk = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a numbered item (1., 2., etc.)
            if line[0].isdigit() and (line[1] == '.' or line[1] == ')'):
                # Save previous risk if exists
                if current_risk:
                    risks.append(' '.join(current_risk).strip())
                    current_risk = []
                # Start new risk (remove number prefix)
                current_risk.append(line.split('.', 1)[1].strip() if '.' in line else line.split(')', 1)[1].strip())
            else:
                # Continue current risk
                current_risk.append(line)
        
        # Add last risk
        if current_risk:
            risks.append(' '.join(current_risk).strip())
        
        return risks


class HuggingFaceRiskSummarizer(RiskSummarizer):
    """Risk summarizer using local Hugging Face models (no API required).
    
    Note: Some models (like Llama) require accepting license agreement on Hugging Face
    and setting HF_TOKEN environment variable. Use models like microsoft/Phi-3-mini-4k-instruct
    or google/flan-t5-large for unrestricted access.
    """
    
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.2-1B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 300,
        prompt_version: str = "v1.0",
        device: Optional[str] = None,
        use_auth_token: Optional[str] = None
    ):
        """Initialize Hugging Face summarizer.
        
        Args:
            model: Model ID from Hugging Face Hub
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            prompt_version: Prompt template version
            device: Device to run on ("cuda", "cpu", or None for auto)
            use_auth_token: Hugging Face token for gated models (or set HF_TOKEN env var)
        """
        super().__init__(model, temperature, max_tokens, prompt_version)
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError(
                "Hugging Face dependencies not installed. "
                "Run: uv sync --all-extras"
            )
        
        import os
        
        # Get auth token from parameter or environment
        auth_token = use_auth_token or os.getenv("HF_TOKEN")
        
        logger.info(f"Loading Hugging Face model: {model}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model,
                token=auth_token,
                trust_remote_code=True
            )
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if device is None else device,
                token=auth_token,
                trust_remote_code=True
            )
        except Exception as e:
            if "gated" in str(e).lower() or "access" in str(e).lower():
                raise ValueError(
                    f"Model {model} requires authentication. Either:\n"
                    f"1. Visit https://huggingface.co/{model} and accept the license\n"
                    f"2. Set HF_TOKEN environment variable with your Hugging Face token\n"
                    f"3. Use an open model like 'meta-llama/Llama-3.2-1B-Instruct'\n"
                    f"Original error: {e}"
                )
            raise
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def summarize(
        self,
        risk_text: str,
        ticker: str,
        company_name: str
    ) -> RiskSummary:
        """Summarize risks using local Hugging Face model."""
        try:
            import torch
            
            prompt = self._build_full_prompt(risk_text, ticker, company_name)
            
            logger.info(
                f"Running local HF model {self.model} for {ticker} "
                f"(prompt v{self.prompt_version})"
            )
            
            # Truncate input to 4096 tokens (temporarily testing with more context)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            input_length = inputs['input_ids'].shape[1]  # Store input length
            inputs = {k: v.to(self.hf_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=300,  # Only need ~200 tokens for 5 bullet points
                    temperature=self.temperature if self.temperature > 0 else 1.0,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,  # Stop when EOS token is generated
                    num_beams=1,  # Disable beam search (use greedy decoding for speed)
                    use_cache=True  # Enable KV cache for faster generation
                )
            
            # Decode ONLY the generated tokens (skip the input prompt)
            generated_tokens = outputs[0][input_length:]
            raw_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            logger.debug(f"Raw model response:\n{raw_response}")
            
            risks = self._parse_response(raw_response)
            
            return RiskSummary(
                risks=risks,
                ticker=ticker,
                company_name=company_name,
                model=self.model,
                prompt_version=self.prompt_version,
                raw_response=raw_response
            )
            
        except ValueError as e:
            logger.error(f"Failed to parse HF response for {ticker}: {e}")
            raise LLMError(f"Response parsing failed: {e}") from e
        except Exception as e:
            logger.error(f"Hugging Face error for {ticker}: {e}")
            raise LLMError(f"Hugging Face summarization failed: {e}") from e
    
    def _build_full_prompt(self, risk_text: str, ticker: str, company_name: str) -> str:
        """Build complete prompt for Hugging Face model.
        
        Smaller models need simpler, more direct prompts with examples.
        """
        # Simplified prompt for smaller models
        simple_prompt = f"""Task: Read the risk factors below and identify the 5 most important risks.

Company: {company_name} ({ticker})

Instructions:
1. List exactly 5 risks
2. Number them 1-5  
3. Each risk should be 1-2 sentences
4. Focus on the biggest threats to the business

Risk Factors:
{risk_text}

Top 5 Risks:
1."""
        
        # Use tokenizer's chat template
        messages = [
            {"role": "user", "content": simple_prompt}
        ]
        
        # Apply the model's native chat template
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def _get_system_prompt(self) -> str:
        """Get system prompt."""
        if self.prompt_version == "v1.0":
            return v1_0.SYSTEM_PROMPT
        else:
            return v1_0.SYSTEM_PROMPT
    
    def _build_prompt(self, risk_text: str, ticker: str, company_name: str) -> str:
        """Build user prompt."""
        if self.prompt_version == "v1.0":
            return v1_0.build_user_prompt(risk_text, ticker, company_name)
        else:
            return v1_0.build_user_prompt(risk_text, ticker, company_name)
    
    def _parse_response(self, response: str) -> List[str]:
        """Parse response into 5 risks."""
        lines = response.strip().split('\n')
        risks = []
        current_risk = []
        
        # Keywords that indicate system prompt leakage (not actual risks)
        system_keywords = [
            'you are a financial analyst',
            'system cutting knowledge date',
            'cutting knowledge date',
            '[inst]',
            '<<sys>>',
            '<</sys>>',
            'your expertise is in',
            'your task is to'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that are clearly system prompt leakage
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in system_keywords):
                continue
            
            # Check if this is a numbered item
            if line[0].isdigit() and (line[1] == '.' or line[1] == ')'):
                if current_risk:
                    risk_text = ' '.join(current_risk).strip()
                    # Final check: skip if this risk contains system keywords
                    if not any(keyword in risk_text.lower() for keyword in system_keywords):
                        risks.append(risk_text)
                    current_risk = []
                current_risk.append(line.split('.', 1)[1].strip() if '.' in line else line.split(')', 1)[1].strip())
            else:
                current_risk.append(line)
        
        if current_risk:
            risk_text = ' '.join(current_risk).strip()
            if not any(keyword in risk_text.lower() for keyword in system_keywords):
                risks.append(risk_text)
        
        if len(risks) != 5:
            logger.error(f"Parsing failed. Found {len(risks)} risks. Raw response:\n{response}")
            raise ValueError(
                f"Expected 5 risks, got {len(risks)}. "
                f"Response may not follow expected format. Found {len(risks)} valid risks after filtering."
            )
        
        return risks


def create_summarizer(
    provider: str = "groq",
    model: Optional[str] = None,
    **kwargs
) -> RiskSummarizer:
    """Factory function to create a risk summarizer.
    
    Args:
        provider: LLM provider ("groq" or "huggingface")
        model: Model name (uses provider default if None)
        **kwargs: Additional arguments passed to summarizer constructor
        
    Returns:
        Configured RiskSummarizer instance
        
    Raises:
        ValueError: If provider is unsupported
        
    Examples:
        >>> summarizer = create_summarizer("groq")  # Free Llama 3.3 70B (default)
        >>> summarizer = create_summarizer("huggingface")  # LOCAL, no API needed
        >>> summarizer = create_summarizer("huggingface", model="meta-llama/Llama-3.2-1B-Instruct")
    """
    provider = provider.lower()
    
    if provider == "groq":
        return GroqRiskSummarizer(
            model=model or "llama-3.3-70b-versatile",
            **kwargs
        )
    elif provider == "huggingface" or provider == "hf":
        return HuggingFaceRiskSummarizer(
            model=model or "meta-llama/Llama-3.2-1B-Instruct",
            **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: 'groq', 'huggingface'"
        )
