"""
Compatibility patch for IndicTransToolkit with newer transformers versions
"""
import logging

logger = logging.getLogger(__name__)

def apply_transformers_compat():
    """Apply compatibility fixes for transformers library"""
    try:
        from transformers.data import data_collator
        
        # Check if the function exists
        if not hasattr(data_collator, 'pad_without_fast_tokenizer_warning'):
            # Add the missing function if it doesn't exist
            def pad_without_fast_tokenizer_warning(*args, **kwargs):
                """Compatibility wrapper for pad function"""
                try:
                    # Try to use pad if it exists
                    if hasattr(data_collator, 'pad'):
                        return data_collator.pad(*args, **kwargs)
                    else:
                        # Fallback implementation
                        return args[0] if args else {}
                except Exception as e:
                    logger.warning(f"pad_without_fast_tokenizer_warning fallback: {e}")
                    return args[0] if args else {}
            
            data_collator.pad_without_fast_tokenizer_warning = pad_without_fast_tokenizer_warning
            logger.info("Applied compatibility patch for pad_without_fast_tokenizer_warning")
            
    except Exception as e:
        logger.warning(f"Could not apply transformers compatibility patch: {e}")

# Apply patch when module is imported
apply_transformers_compat()


