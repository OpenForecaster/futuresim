from .base import DataFetcher, CachedQuestion
import requests
import json
from datetime import datetime, date
import time

class MetaculusFetcher(DataFetcher):
    """
    Fetcher for Metaculus API.
    Handles 'metaculus_binary' and 'metaculus_mcq'.
    """
    
    BASE_URL = "https://www.metaculus.com/api2/questions/"
    
    def __init__(self, dataset_path: str = None, split: str = "train", type_filter: str = "binary", min_forecasters: int = 10):
        super().__init__(dataset_path, split)
        self.type_filter = type_filter # 'binary' or 'multiple_choice'
        self.min_forecasters = min_forecasters
        
    @property
    def source_name(self) -> str:
        return f"metaculus_{self.type_filter}"
        
    def get_prompt_context(self) -> str:
        return f"""
## METACULUS DATA SOURCE
This question comes from Metaculus. 
Background info provides context and current crowd forecast statistics if available.
"""

    def fetch_new(self, start_date: date, end_date: date) -> list[CachedQuestion]:
        """Fetch questions resolving between start and end date."""
        print(f"Fetching {self.source_name} resolving between {start_date} and {end_date}...")
        print(f"Filters: status='resolved', nr_forecasters >= {self.min_forecasters}")
        
        limit = 100
        offset = 0
        questions = []
        
        while True:
            params = {
                "limit": limit,
                "offset": offset,
                "resolution_time__gt": start_date.isoformat(),
                "resolution_time__lt": end_date.isoformat(),
                "type": self.type_filter
            }
            
            try:
                resp = requests.get(self.BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
                results = data.get('results', [])
                
                if not results:
                    break
                    
                if not results:
                    break
                    
                for item in results:
                    q = self._parse_question(item)
                    if q:
                        questions.append(q)
                
                if not data.get('next'):
                    break
                    
                offset += limit
                time.sleep(0.5) # Rate limit politeness
                print(f"  Fetched {len(questions)} questions so far... (offset {offset})")
                
            except Exception as e:
                print(f"Error fetching page at offset {offset}: {e}")
                break
                
        print(f"Total fetched: {len(questions)}")
        return questions
        
    def _parse_question(self, item: dict) -> CachedQuestion:
        """Parse API response item to CachedQuestion."""
        # Date handling
        res_time_str = item.get('actual_resolve_time') or item.get('scheduled_resolve_time')
        if not res_time_str:
            return None
            
        try:
            res_date = datetime.fromisoformat(res_time_str.replace("Z", "+00:00")).date()
        except ValueError:
            return None
            
        # Get question details from nested dict
        q_details = item.get('question', {})
        q_type = q_details.get('type')
        
        # Extract options
        options = None
        possibilities = q_details.get('possibilities') or {}
        
        if self.type_filter == 'multiple_choice':
            if q_type != 'multiple_choice':
                return None
                
            options = possibilities.get('values', [])
            if not options and 'labels' in possibilities:
                options = possibilities['labels']
            
            if not options:
                return None

        elif self.type_filter == 'binary':
            # Metaculus sometimes labels binary as 'forecast' with type='binary' in question
            if q_type != 'binary':
                 return None
            options = ["Yes", "No"]
            
        # Ground truth
        resolution = q_details.get('resolution')
        ground_truth = ""
        
        if resolution is not None:
            if self.type_filter == 'binary':
                # Resolution can be:
                # - String: "yes", "no" (newer API)
                # - Numeric: 0.0 (No), 1.0 (Yes) (older API)
                if isinstance(resolution, str):
                    if resolution.lower() == 'yes':
                        ground_truth = "Yes"
                    elif resolution.lower() == 'no':
                        ground_truth = "No"
                elif isinstance(resolution, (int, float)):
                    if resolution >= 0.5:
                        ground_truth = "Yes"
                    else:
                        ground_truth = "No"
            elif self.type_filter == 'multiple_choice' and options:
                 # Check if resolution is index or value or string
                 if isinstance(resolution, str):
                     # Direct option match
                     if resolution in options:
                         ground_truth = resolution
                     else:
                         # Try case-insensitive
                         for opt in options:
                             if opt.lower() == resolution.lower():
                                 ground_truth = opt
                                 break
                 elif isinstance(resolution, int) and 0 <= resolution < len(options):
                     ground_truth = options[resolution]
                 elif isinstance(resolution, float):
                     idx = int(resolution)
                     if 0 <= idx < len(options):
                         ground_truth = options[idx]

        return CachedQuestion(
            qid=str(item.get('id')),
            title=item.get('title', ''),
            background=item.get('description', '') or q_details.get('description', ''),
            resolution_criteria=q_details.get('resolution_criteria', ''),
            answer_type=self.type_filter,
            resolution_date=res_date,
            ground_truth_answer=ground_truth,
            options=options,
            source=self.source_name,
            metadata={
                'source': 'metaculus',
                'status': item.get('status'),
                'resolved': item.get('resolved'),
                'nr_forecasters': item.get('nr_forecasters'),
                'forecasts_count': item.get('forecasts_count'),
                'votes': item.get('votes')
            }
        )
