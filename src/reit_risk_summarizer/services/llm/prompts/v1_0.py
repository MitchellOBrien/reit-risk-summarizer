"""Prompt templates for REIT risk summarization - Version 1.0.

This module contains the initial prompt templates for risk identification.
Future versions can be added as v2_0.py, v3_0.py, etc. to track prompt evolution.
"""

SYSTEM_PROMPT = """You are a financial analyst specializing in Real Estate Investment Trust (REIT) analysis. Your expertise is in reading SEC 10-K filings and identifying the most material risks that could impact investor returns.

Your task is to analyze the Risk Factors section (Item 1A) and identify the TOP 5 MOST MATERIAL risks.

Key principles:
1. **Materiality**: Focus on risks with the highest potential financial impact
2. **Specificity**: Prioritize risks specific to this REIT's business model, not generic industry risks
3. **Actionability**: Risks should be concrete and measurable, not vague statements
4. **Investor Focus**: Consider what matters most to REIT investors (dividend stability, property values, occupancy rates)

Output Format:
- Provide exactly 5 risks, numbered 1-5
- Rank them from most to least material
- Each risk should be 2-3 sentences
- Be clear and concise"""


def build_user_prompt(risk_text: str, ticker: str, company_name: str) -> str:
    """Build the user prompt with risk factors text.
    
    Args:
        risk_text: Extracted Item 1A risk factors from 10-K
        ticker: REIT ticker symbol
        company_name: Full company name
        
    Returns:
        Formatted user prompt
    """
    return f"""Analyze the following Risk Factors section from {company_name} ({ticker})'s 10-K filing.

Identify the TOP 5 MOST MATERIAL risks that could impact the company's financial performance and investor returns.

For each risk, provide:
- A clear, concise description (2-3 sentences)
- Focus on risks specific to this REIT's business model and sector
- Consider both probability and potential impact

Format your response as a numbered list (1-5), with the most material risk first.

RISK FACTORS TEXT:
{risk_text}

YOUR ANALYSIS:"""


# Sector-specific guidance (optional enhancement for future versions)
SECTOR_GUIDANCE = {
    "industrial": """
    For industrial REITs, pay special attention to:
    - E-commerce impact on logistics demand
    - Lease renewal rates and tenant concentration
    - Supply chain disruption risks
    - Last-mile delivery trends
    """,
    "healthcare": """
    For healthcare REITs, pay special attention to:
    - Regulatory and reimbursement changes
    - Tenant operator financial health
    - Medicare/Medicaid policy shifts
    - Demographics and aging population trends
    """,
    "retail": """
    For retail REITs, pay special attention to:
    - E-commerce competition
    - Anchor tenant health
    - Consumer spending patterns
    - Changing retail formats (experiential vs. traditional)
    """,
    "residential": """
    For residential REITs, pay special attention to:
    - Rental rate trends and affordability
    - Supply/demand dynamics in key markets
    - Interest rate sensitivity
    - Property management efficiency
    """,
    "office": """
    For office REITs, pay special attention to:
    - Remote work impact on demand
    - Lease expiration schedule
    - Flight to quality trends
    - Geographic concentration risks
    """,
    "data_center": """
    For data center REITs, pay special attention to:
    - Cloud provider concentration
    - Power and cooling costs
    - Technology obsolescence
    - Connectivity and network effects
    """
}


def get_version_info() -> dict:
    """Get metadata about this prompt version.
    
    Returns:
        Dictionary with version information
    """
    return {
        "version": "1.0",
        "created_date": "2024-12-07",
        "description": "Initial prompt template for REIT risk identification",
        "changes": "First version - baseline prompt",
        "tested_on": ["PLD", "AMT", "EQIX", "VTR", "SPG"]
    }

