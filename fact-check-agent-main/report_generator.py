import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_report_df(results: list) -> pd.DataFrame:
    """
    Converts verification results list into a Pandas DataFrame.
    
    Args:
        results (list): A list of dictionaries containing keys:
                        'claim', 'status', 'confidence', 'reason', 'correct_fact', 'evidence'
                        
    Returns:
        pd.DataFrame: Formatted DataFrame ready for display and export.
    """
    rows = []
    
    for item in results:
        claim = item.get("claim", "")
        status = item.get("status", "")
        confidence = item.get("confidence", 0)
        reason = item.get("reason", "")
        correct_fact = item.get("correct_fact", "")
        
        # Consolidate source URLs
        evidence_list = item.get("evidence", [])
        source_urls = [e.get("url") for e in evidence_list if e.get("url")]
        sources_str = "; ".join(source_urls) if source_urls else "No sources found"
        
        rows.append({
            "Claim": claim,
            "Status": status,
            "Confidence (%)": confidence,
            "Correct Fact": correct_fact if correct_fact else "N/A (Verified)",
            "Verification Reason": reason,
            "Sources": sources_str
        })
        
    df = pd.DataFrame(rows)
    logger.info(f"Generated report DataFrame with {len(df)} rows.")
    return df

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """
    Converts a DataFrame to a CSV bytes object (UTF-8 encoded) for download.
    
    Args:
        df (pd.DataFrame): The results DataFrame.
        
    Returns:
        bytes: CSV file encoded as bytes.
    """
    try:
        csv_str = df.to_csv(index=False)
        return csv_str.encode("utf-8")
    except Exception as e:
        logger.error(f"Error converting DataFrame to CSV: {str(e)}")
        return b""
