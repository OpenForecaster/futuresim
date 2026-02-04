from .base import DataFetcher

def get_fetcher(dataset: str, dataset_path: str = None, dataset_cache: str = None, split: str = "train") -> DataFetcher:
    """Factory to get the correct data fetcher."""
    
    if "openforesight" in dataset.lower():
        from .openforesight import OpenForesightFetcher
        return OpenForesightFetcher(dataset_path, split)
        
    if "metaculus" in dataset.lower():
        from .metaculus import MetaculusFetcher
        type_filter = "multiple_choice" if "mcq" in dataset.lower() else "binary"
        return MetaculusFetcher(dataset_path, split, type_filter)
        
    raise ValueError(f"Unknown dataset type: {dataset}")
