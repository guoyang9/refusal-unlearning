import re
from typing import Dict


def split_gpt_oss_response(response: str) -> Dict[str, str]:
    """Split the response into analysis and final content"""
    # Find the analysis part (everything before "assistantfinal")
    analysis_match = re.search(r'^(.*?)assistantfinal', response, flags=re.DOTALL)

    if analysis_match:
        analysis_content = analysis_match.group(1).strip()
        # Remove "analysis" prefix if present
        analysis_content = re.sub(r'^analysis\s*', '', analysis_content)
        
        # Find the final content (everything after "assistantfinal")
        final_content = re.sub(r'^.*?assistantfinal\s*', '', response, flags=re.DOTALL).strip()
    else:
        # If no "assistantfinal" found, check for just "analysis" prefix
        if response.startswith('analysis'):
            analysis_content = response.strip()
            final_content = ""
        else:
            # No clear separation, treat entire response as final content
            analysis_content = ""
            final_content = response.strip()

    return {
        "analysis_content": analysis_content,
        "final_content": final_content
    }