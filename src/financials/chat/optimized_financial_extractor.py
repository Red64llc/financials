import re
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
import logging

logger = logging.getLogger(__name__)

class OptimizedFinancialExtractor:
    """Optimized Financial Data Extractor
    
    This class implements an optimized financial data extractor that uses vector search to find relevant chunks of text
    and extract financial data from them.
    
    Args:
        embedding_model_name: Name of the sentence transformer model to use
    Returns:
        financial_data: DataFrame containing the extracted financial data
    Usage:
        extractor = OptimizedFinancialExtractor()
        financial_data = extractor.extract()

    """
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.load_models()
        self.revenue_data = []
        self.output_dir = Path("./output")
        self.output_dir.mkdir(exist_ok=True)
    
    def load_models(self):
        """Load specialized financial models once"""
        logger.info("Loading specialized financial models...")
        self.finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.finbert = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        self.finbert_nlp = pipeline("sentiment-analysis", model=self.finbert, tokenizer=self.finbert_tokenizer)
        self.ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.ner_pipeline = pipeline("ner", model=self.ner_model, tokenizer=self.ner_tokenizer, aggregation_strategy="simple")
        logger.info("Models loaded successfully")
    
    def extract(self, chunks):
        if chunks is None:
            raise ValueError("No chunks provided")
        
        unique_chunks = self._deduplicate_chunks(chunks)
        logger.info(f"Found {len(unique_chunks)} unique financial chunks")
        
        for chunk in unique_chunks:
            chunk_text = chunk.page_content
            metadata = chunk.metadata
            self._process_chunk(chunk_text, metadata)
        
        return self._analyze_revenue_data()
    
    def _deduplicate_chunks(self, chunks):
        """Deduplicate chunks based on content hash"""
        seen_contents = set()
        unique_chunks = []
        
        for chunk in chunks:
            content_hash = hash(chunk.page_content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _process_chunk(self, text, metadata):
        """Process a single text chunk for financial data extraction"""
        # Check if chunk is related to financial information using FinBERT
        if len(text.split()) < 6:  # Only process substantial chunks
            return
        try:
            # Use FinBERT to identify financial context
            finbert_result = self.finbert_nlp(text[:512])  # Limit to model's max length
            
            # Only process chunks identified as financial in nature
            if any(result['label'] in ['neutral', 'positive', 'negative'] for result in finbert_result):
                # Search for revenue-related terms
                if re.search(r'\b(revenue|sales|turnover|income|earnings)\b', text.lower()):
                    # Extract monetary amounts
                    monetary_amounts = self._extract_monetary_amounts(text)
                    
                    # Extract time periods
                    time_periods = self._extract_time_periods(text)
                    
                    # If we have both money values and time periods, associate them
                    if monetary_amounts and time_periods:
                        for amount_info in monetary_amounts:
                            for period in time_periods:
                                # Calculate confidence based on proximity
                                confidence = self._calculate_proximity_confidence(
                                    text, amount_info['text'], period['text']
                                )
                                
                                # Source from metadata
                                source = metadata.get('source', 'unknown')
                                
                                self.revenue_data.append({
                                    'source': source,
                                    'period': period['normalized'],
                                    'value': amount_info['value'],
                                    'unit': amount_info['unit'],
                                    'confidence': confidence,
                                    'context': text[:100] + '...' if len(text) > 100 else text
                                })
        except Exception as e:
            logger.error(f"Error analyzing chunk with FinBERT: {e}")
    
    def _extract_monetary_amounts(self, text):
        """Extract monetary amounts from text"""
        # Implementation of monetary amount extraction
        # This is the same logic from the original FinancialDataExtractor
        amounts = []
        
        # Pattern for numbers with optional thousand separators and decimal places
        number_pattern = r'(\$|€|£|¥)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(million|billion|m|b|k|thousand)?'
        
        matches = re.finditer(number_pattern, text, re.IGNORECASE)
        
        for match in matches:
            try:
                currency_symbol = match.group(1) or ''
                value_str = match.group(2).replace(',', '')
                multiplier_str = match.group(3) or ''
                
                value = float(value_str)
                
                # Apply multiplier
                multiplier = 1
                if multiplier_str.lower() in ['million', 'm']:
                    multiplier = 1000000
                elif multiplier_str.lower() in ['billion', 'b']:
                    multiplier = 1000000000
                elif multiplier_str.lower() in ['thousand', 'k']:
                    multiplier = 1000
                
                value *= multiplier
                
                # Determine unit/currency
                unit = 'Unknown'
                if currency_symbol:
                    if currency_symbol == '$':
                        unit = 'USD'
                    elif currency_symbol == '€':
                        unit = 'EUR'
                    elif currency_symbol == '£':
                        unit = 'GBP'
                    elif currency_symbol == '¥':
                        unit = 'JPY'
                else:
                    # Look for currency mentions near the number
                    context = text[max(0, match.start() - 30):min(len(text), match.end() + 30)].lower()
                    if 'dollar' in context or 'usd' in context:
                        unit = 'USD'
                    elif 'euro' in context or 'eur' in context:
                        unit = 'EUR'
                    elif 'pound' in context or 'gbp' in context:
                        unit = 'GBP'
                    elif 'yen' in context or 'jpy' in context:
                        unit = 'JPY'
                
                # Check if this is likely a revenue figure
                context = text[max(0, match.start() - 50):min(len(text), match.end() + 50)].lower()
                revenue_indicators = ['revenue', 'sales', 'turnover', 'income', 'earnings']
                is_revenue = any(indicator in context for indicator in revenue_indicators)
                
                if is_revenue:
                    amounts.append({
                        'text': match.group(0),
                        'value': value,
                        'unit': unit,
                        'start_idx': match.start(),
                        'end_idx': match.end()
                    })
            except Exception as e:
                logger.error(f"Error parsing monetary amount: {e}")
        
        return amounts
    
    def _extract_time_periods(self, text):
        """Extract time periods from text"""
        # Implementation for time period extraction 
        periods = []
        
        # Patterns for different time period formats
        patterns = [
            # Quarters: Q1 2023, 1Q 2023, etc.
            r'(?:Q([1-4])\s*(\d{4})|([1-4])Q\s*(\d{4}))',
            
            # Fiscal years: FY2023, FY 2023, Fiscal Year 2023, etc.
            r'(?:FY|Fiscal Year|Fiscal)\s*(\d{4})',
            
            # Years: 2023, etc.
            r'(?<![A-Za-z0-9])(\d{4})(?![A-Za-z0-9\-])',
            
            # Month Year: January 2023, Jan 2023, etc.
            r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,\s]+(\d{4})'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                try:
                    normalized_period = ''
                    match_text = match.group(0)
                    
                    # Normalize quarters
                    q_match = re.search(r'Q([1-4])\s*(\d{4})|([1-4])Q\s*(\d{4})', match_text)
                    if q_match:
                        quarter = q_match.group(1) or q_match.group(3)
                        year = q_match.group(2) or q_match.group(4)
                        normalized_period = f"Q{quarter} {year}"
                    
                    # Normalize fiscal years
                    fy_match = re.search(r'(?:FY|Fiscal Year|Fiscal)\s*(\d{4})', match_text)
                    if fy_match:
                        normalized_period = f"FY{fy_match.group(1)}"
                    
                    # Normalize months
                    month_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,\s]+(\d{4})', match_text)
                    if month_match:
                        month = month_match.group(1)
                        year = month_match.group(2)
                        # Map abbreviations to full names
                        month_map = {
                            'Jan': 'January', 'Feb': 'February', 'Mar': 'March',
                            'Apr': 'April', 'May': 'May', 'Jun': 'June',
                            'Jul': 'July', 'Aug': 'August', 'Sep': 'September',
                            'Oct': 'October', 'Nov': 'November', 'Dec': 'December'
                        }
                        if month in month_map:
                            month = month_map[month]
                        normalized_period = f"{month} {year}"
                    
                    # Standalone years
                    if not normalized_period and re.match(r'^\d{4}$', match_text):
                        normalized_period = match_text
                    
                    # Check if this looks like a revenue reporting period
                    context = text[max(0, match.start() - 50):min(len(text), match.end() + 50)].lower()
                    revenue_indicators = ['revenue', 'sales', 'turnover', 'income', 'earnings']
                    is_revenue_period = any(indicator in context for indicator in revenue_indicators)
                    
                    if is_revenue_period and normalized_period:
                        periods.append({
                            'text': match_text,
                            'normalized': normalized_period,
                            'start_idx': match.start(),
                            'end_idx': match.end()
                        })
                
                except Exception as e:
                    logger.error(f"Error parsing time period: {e}")
        
        return periods
    
    def _calculate_proximity_confidence(self, text, amount_text, period_text):
        """Calculate confidence based on proximity of amount and period in text"""
        try:
            # Find all occurrences of amount_text and period_text in the text
            amount_idx = text.find(amount_text)
            period_idx = text.find(period_text)
            
            if amount_idx == -1 or period_idx == -1:
                return 0.5  # Default confidence
            
            # Calculate distance
            distance = abs(amount_idx - period_idx)
            
            # Assign confidence based on distance
            if distance < 50:
                return 0.9  # High confidence for close proximity
            elif distance < 150:
                return 0.7  # Medium confidence
            else:
                return 0.5  # Lower confidence for greater distances
        except Exception as e:
            logger.error(f"Error calculating proximity confidence: {e}")
            return 0.5
    
    def _analyze_revenue_data(self):
        """Analyze extracted revenue data to calculate growth rates"""
        if not self.revenue_data:
            logger.error("No revenue data found")
            return None
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.revenue_data)
        
        # Save raw extracted data
        df.to_csv(self.output_dir / "revenue_data_raw.csv", index=False)
        
        # Filter by confidence 
        df_high_conf = df[df['confidence'] >= 0.7]
        
        # Group by period and calculate weighted average based on confidence
        df['weighted_value'] = df['value'] * df['confidence']
        grouped = df.groupby('period').agg({
            'weighted_value': 'sum',
            'confidence': 'sum',
            'unit': lambda x: x.value_counts().index[0],
            'source': lambda x: ', '.join(set(x))
        }).reset_index()
        
        # Calculate final value using weighted average
        grouped['value'] = grouped['weighted_value'] / grouped['confidence']
        grouped.drop('weighted_value', axis=1, inplace=True)
        
        # Sort by period
        try:
            # Try to convert periods to datetime for proper sorting
            grouped['period_for_sort'] = grouped['period'].apply(self._period_to_sortable_date)
            grouped = grouped.sort_values('period_for_sort').drop('period_for_sort', axis=1)
        except:
            # If conversion fails, sort alphanumerically
            grouped = grouped.sort_values('period')
        
        # Calculate growth rates
        grouped['prev_value'] = grouped['value'].shift(1)
        grouped['growth_rate'] = (grouped['value'] - grouped['prev_value']) / grouped['prev_value']
        
        # Save processed data
        grouped.to_csv(self.output_dir / "revenue_data_analyzed.csv", index=False)
        
        logger.debug(f"Processed {len(df)} revenue data points across {len(grouped)} time periods")
        return grouped
    
    def _period_to_sortable_date(self, period):
        """Convert period string to a sortable date format"""
        # Implementation for sorting periods correctly
        try:
            # Handle quarters (Q1 2023, etc.)
            q_match = re.search(r'Q([1-4])\s*(\d{4})', period)
            if q_match:
                quarter = int(q_match.group(1))
                year = int(q_match.group(2))
                month = (quarter - 1) * 3 + 2  # Middle month of quarter
                return f"{year}-{month:02d}-15"
            
            # Handle fiscal years (FY2023, etc.)
            fy_match = re.search(r'FY(\d{4})', period)
            if fy_match:
                year = int(fy_match.group(1))
                return f"{year}-06-30"  # Midpoint of fiscal year
            
            # Handle years (2023, etc.)
            year_match = re.search(r'(\d{4})', period)
            if year_match:
                return f"{year_match.group(1)}-06-30"  # Midpoint of year
            
            # Handle month abbreviations (Jan 2023, etc.)
            month_year_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})', period)
            if month_year_match:
                month_str = month_year_match.group(1)
                year = int(month_year_match.group(2))
                month_map = {
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                }
                month = month_map.get(month_str, 1)
                return f"{year}-{month:02d}-15"
        except:
            pass
        
        # Default fallback
        return period
