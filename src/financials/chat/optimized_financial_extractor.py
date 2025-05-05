import re
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import pipeline
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
        self._initialize_financial_taxonomy()
        self.financial_data = []
    
    def load_models(self):
        """Load specialized financial models once"""
        logger.info("Loading specialized financial models...")
        # Load FinBERT for financial context understanding
        self.finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.finbert = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        self.finbert_nlp = pipeline("sentiment-analysis", model=self.finbert, tokenizer=self.finbert_tokenizer)
        
        # Load NER model for entity recognition
        self.ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.ner_pipeline = pipeline("ner", model=self.ner_model, tokenizer=self.ner_tokenizer, aggregation_strategy="simple")

        # Zero-shot classifier for financial metric classification
        # We'll use this to categorize financial entities when patterns are ambiguous
        try:
            self.zero_shot = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli"
            )
        except:
            self.zero_shot = None
            logger.warning("Zero-shot classifier not available. Entity classification may be less accurate.")

        logger.info("Specialized financial models loaded successfully")
    
    def _initialize_financial_taxonomy(self):
        """Initialize the taxonomy of financial entities and relationships"""
        # Primary financial metrics
        self.financial_metrics = {
            # Revenue metrics
            'revenue': {
                'patterns': [
                    r'\brevenue\b', r'\bsales\b', r'\bturnover\b', r'\btop.line\b', 
                    r'\bgross revenue\b', r'\bnet revenue\b', r'\btotal revenue\b'
                ],
                'subtypes': ['total revenue', 'product revenue', 'service revenue', 'subscription revenue'],
                'relations': ['segment', 'region', 'product', 'period'],
                'growth_applicable': True
            },
            
            # Cost metrics
            'cost': {
                'patterns': [
                    r'\bcost\b', r'\bexpense\b', r'\bexpenditure\b', r'\bcosts\b', 
                    r'\bcost of (sales|revenue|goods)\b', r'\bopex\b', r'\bcapex\b'
                ],
                'subtypes': ['cost of goods sold', 'operating expenses', 'marketing expenses', 
                            'research and development', 'SG&A', 'capex'],
                'relations': ['segment', 'region', 'product', 'period'],
                'growth_applicable': True
            },
            
            # Profit metrics
            'profit': {
                'patterns': [
                    r'\bprofit\b', r'\bearnings\b', r'\bincome\b', r'\bnet income\b', 
                    r'\bebit\b', r'\bebitda\b', r'\boperating (income|profit)\b', 
                    r'\bgross profit\b', r'\bnet profit\b', r'\bprofit margin\b'
                ],
                'subtypes': ['gross profit', 'operating profit', 'net profit', 'EBIT', 'EBITDA'],
                'relations': ['segment', 'region', 'product', 'period'],
                'growth_applicable': True
            },
            
            # Margin metrics
            'margin': {
                'patterns': [
                    r'\bmargin\b', r'\bprofit margin\b', r'\bgross margin\b', 
                    r'\boperating margin\b', r'\bnet margin\b'
                ],
                'subtypes': ['gross margin', 'operating margin', 'net margin', 'contribution margin'],
                'relations': ['segment', 'region', 'product', 'period'],
                'growth_applicable': True
            },
            
            # Growth metrics
            'growth': {
                'patterns': [
                    r'\bgrowth\b', r'\bincrease\b', r'\bgrew by\b', r'\bgrowth rate\b',
                    r'\byear.over.year\b', r'\byoy\b', r'\bcompound annual growth\b', 
                    r'\bcagr\b', r'\bq.o.q\b'
                ],
                'subtypes': ['year-over-year', 'quarter-over-quarter', 'CAGR'],
                'relations': ['metric', 'segment', 'region', 'product', 'period'],
                'growth_applicable': False  # Growth of growth doesn't make sense
            },
            
            # Cash flow metrics
            'cash_flow': {
                'patterns': [
                    r'\bcash flow\b', r'\boperating cash flow\b', r'\bfree cash flow\b', 
                    r'\bcash from operations\b', r'\bcash generated\b'
                ],
                'subtypes': ['operating cash flow', 'free cash flow', 'cash from investing', 
                           'cash from financing'],
                'relations': ['period'],
                'growth_applicable': True
            },
            
            # Balance sheet metrics
            'balance_sheet': {
                'patterns': [
                    r'\bassets\b', r'\bliabilities\b', r'\bequity\b', r'\bdebt\b', 
                    r'\bcash(\sand\scash\sequivalents)?\b', r'\binventory\b', r'\bpp&e\b',
                    r'\baccounts receivable\b', r'\baccounts payable\b'
                ],
                'subtypes': ['total assets', 'current assets', 'non-current assets', 
                           'total liabilities', 'current liabilities', 'long-term debt'],
                'relations': ['period'],
                'growth_applicable': True
            },
            
            # Performance metrics
            'performance': {
                'patterns': [
                    r'\broi\b', r'\broe\b', r'\broa\b', r'\broi[ce]\b', r'\beps\b', 
                    r'\bearnings per share\b', r'\bpe ratio\b', r'\bprice.to.earnings\b'
                ],
                'subtypes': ['ROI', 'ROE', 'ROA', 'ROIC', 'EPS', 'diluted EPS'],
                'relations': ['period'],
                'growth_applicable': True
            },
            
            # Market metrics
            'market': {
                'patterns': [
                    r'\bmarket share\b', r'\bcustomer acquisitions?\b', r'\buser base\b', 
                    r'\bactive users\b', r'\bcustomer retention\b', r'\bchurn\b'
                ],
                'subtypes': ['market share', 'customer acquisition', 'retention rate', 'churn rate'],
                'relations': ['segment', 'region', 'product', 'period'],
                'growth_applicable': True
            }
        }
        
        # Build a combined pattern for quick first-pass detection
        all_patterns = []
        for metric_type in self.financial_metrics:
            all_patterns.extend(self.financial_metrics[metric_type]['patterns'])
        
        self.combined_financial_pattern = re.compile('|'.join(all_patterns), re.IGNORECASE)
        
        # Business segments - common industry categories
        # This would be customized for the specific company
        self.business_segments = [
            'cloud services', 'software', 'hardware', 'services', 'consulting',
            'north america', 'emea', 'asia pacific', 'latin america',
            'enterprise', 'consumer', 'small business', 'government'
        ]

    def extract(self, chunks):
        if chunks is None:
            raise ValueError("No chunks provided")
        
        unique_chunks = self._deduplicate_chunks(chunks)
        logger.info(f"Found {len(unique_chunks)} unique financial chunks")
        logger.info(f"Processing chunks...")
        for chunk in unique_chunks:
            chunk_text = chunk.page_content
            metadata = chunk.metadata
            self._process_chunk(chunk_text, metadata)
        
        return self._analyze_financial_data()
    
    def _deduplicate_chunks(self, chunks):
        return chunks
        # """Deduplicate chunks based on content hash"""
        # seen_contents = set()
        # unique_chunks = []
        
        # for chunk in chunks:
        #     content_hash = hash(chunk.page_content)
        #     if content_hash not in seen_contents:
        #         seen_contents.add(content_hash)
        #         unique_chunks.append(chunk)
        
        # return unique_chunks
    
    def _process_chunk(self, text, metadata):
        """Process a single text chunk for financial data extraction"""
        # Check if chunk is related to financial information using FinBERT
        if len(text.split()) < 6:  # Only process substantial chunks
            return
        try:
            logger.info(f"Processing chunk: {text}")

            # First, check if there are any financial metric patterns in the text
            # This is a fast preliminary check before running the model
            if not self.combined_financial_pattern.search(text.lower()):
                logger.info("No financial metric patterns found in chunk")
                return
                
            # Use FinBERT to identify financial context
            finbert_result = self.finbert_nlp(text[:512])  # Limit to model's max length
            logger.info(f"FinBERT result: {finbert_result}")
            
            # Extract monetary amounts
            monetary_amounts = self._extract_monetary_amounts(text)
            logger.info(f"Extracted monetary amounts: {monetary_amounts}")
            
            # Extract time periods
            time_periods = self._extract_time_periods(text)
            logger.info(f"Extracted time periods: {time_periods}")
            
            entities = self._identify_financial_entities(text)
            logger.info(f"Extracted entities: {entities}")

            # Associate monetary amounts with financial entities and time periods
            self._associate_values_with_entities(
                text, 
                entities, 
                monetary_amounts, 
                time_periods, 
                metadata
            )

        except Exception as e:
            logger.error(f"Error analyzing chunk with FinBERT: {e}")
    
    def _extract_monetary_amounts(self, text):
        """Extract monetary amounts from text"""
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
                
                amounts.append({
                    'text': match.group(0),
                    'value': value,
                    'unit': unit,
                    'position': match.start()
                })
            except ValueError:
                continue
        
        return amounts
    
    def _extract_time_periods(self, text):
        """Extract time periods from text"""
        periods = []
        
        # Pattern for quarters (Q1 2023, etc.)
        quarter_pattern = r'Q([1-4])\s*(\d{4}|\d{2})'
        # Pattern for fiscal years (FY2023, fiscal year 2023, etc.)
        fiscal_pattern = r'(?:FY|fiscal year|financial year)\s*(\d{4}|\d{2})'
        # Pattern for years
        year_pattern = r'\b(20\d{2}|19\d{2})\b'
        # Pattern for month-year (Jan 2023, January 2023, etc.)
        month_year_pattern = r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})'
        
        # Find quarters
        for match in re.finditer(quarter_pattern, text, re.IGNORECASE):
            quarter = match.group(1)
            year = match.group(2)
            # Standardize two-digit years
            if len(year) == 2:
                year = '20' + year
            
            periods.append({
                'text': match.group(0),
                'normalized': f"Q{quarter} {year}",
                'position': match.start()
            })
        
        # Find fiscal years
        for match in re.finditer(fiscal_pattern, text, re.IGNORECASE):
            year = match.group(1)
            # Standardize two-digit years
            if len(year) == 2:
                year = '20' + year
                
            periods.append({
                'text': match.group(0),
                'normalized': f"FY{year}",
                'position': match.start()
            })
        
        # Find years (if not already captured in quarters or fiscal years)
        if not periods:
            for match in re.finditer(year_pattern, text):
                periods.append({
                    'text': match.group(0),
                    'normalized': match.group(1),
                    'position': match.start()
                })
        
        # Find month-year combinations
        for match in re.finditer(month_year_pattern, text, re.IGNORECASE):
            month = match.group(1)[:3]  # Take first 3 letters
            year = match.group(2)
            periods.append({
                'text': match.group(0),
                'normalized': f"{month} {year}",
                'position': match.start()
            })
        
        return periods
    
    def _identify_financial_entities(self, text):
        """Identify financial entities in the text"""
        entities = []
        
        # Check for each financial metric type
        for metric_type, metric_info in self.financial_metrics.items():
            # Create a combined pattern for this metric type
            metric_pattern = re.compile('|'.join(metric_info['patterns']), re.IGNORECASE)
            
            # Find all matches for this metric type
            for match in metric_pattern.finditer(text.lower()):
                entity_text = match.group(0)
                
                # Determine the specific subtype by looking at surrounding text
                context = text[max(0, match.start() - 50):min(len(text), match.end() + 50)].lower()
                subtype = self._determine_entity_subtype(entity_text, context, metric_info['subtypes'])
                
                # Look for related dimensions (segments, products, regions)
                dimensions = self._extract_entity_dimensions(context)
                
                entities.append({
                    'type': metric_type,
                    'subtype': subtype,
                    'text': entity_text,
                    'position': match.start(),
                    'dimensions': dimensions,
                    'growth_applicable': metric_info['growth_applicable']
                })
                
        # If no entities found but there's clearly financial content, use fallback
        if not entities and self.combined_financial_pattern.search(text.lower()):
            # Use zero-shot classification if available
            if self.zero_shot:
                # Create a list of candidate labels from our financial taxonomy
                candidate_labels = list(self.financial_metrics.keys())
                
                # Run zero-shot classification
                result = self.zero_shot(text, candidate_labels, multi_label=True)
                
                # Use the top prediction if it has reasonable confidence
                if result['scores'][0] > 0.5:
                    top_label = result['labels'][0]
                    entities.append({
                        'type': top_label,
                        'subtype': 'unknown',
                        'text': 'financial metric',  # Generic placeholder
                        'position': 0,  # Unknown position
                        'dimensions': self._extract_entity_dimensions(text),
                        'growth_applicable': self.financial_metrics[top_label]['growth_applicable']
                    })
            else:
                # Fallback to generic financial entity
                entities.append({
                    'type': 'unknown',
                    'subtype': 'unknown',
                    'text': 'financial metric',
                    'position': 0,
                    'dimensions': self._extract_entity_dimensions(text),
                    'growth_applicable': True
                })
        
        return entities
    
    def _determine_entity_subtype(self, entity_text, context, subtypes):
        """Determine the specific subtype of a financial entity"""
        # Check if any subtype is explicitly mentioned
        for subtype in subtypes:
            if subtype.lower() in context:
                return subtype
        
        # If no explicit subtype is found but entity contains specific keywords
        entity_lower = entity_text.lower()
        if 'gross' in entity_lower:
            return 'gross ' + entity_lower.split('gross')[-1].strip()
        elif 'net' in entity_lower:
            return 'net ' + entity_lower.split('net')[-1].strip()
        elif 'operating' in entity_lower:
            return 'operating ' + entity_lower.split('operating')[-1].strip()
        
        # Default to the general entity type
        return entity_text
    
    def _extract_entity_dimensions(self, context):
        """Extract dimensions like segments, products, regions"""
        dimensions = {}
        
        # Look for business segments
        for segment in self.business_segments:
            if segment.lower() in context.lower():
                segment_type = self._categorize_segment(segment)
                if segment_type in dimensions:
                    dimensions[segment_type].append(segment)
                else:
                    dimensions[segment_type] = [segment]
        
        # Look for specific product mentions
        product_pattern = r'(\w+\s)?(product|service|platform|solution|application)s?'
        for match in re.finditer(product_pattern, context, re.IGNORECASE):
            product = match.group(0)
            if 'product' not in dimensions:
                dimensions['product'] = []
            dimensions['product'].append(product)
        
        return dimensions
    
    def _categorize_segment(self, segment):
        """Categorize a segment as region, business unit, customer type, etc."""
        regions = ['north america', 'europe', 'asia', 'emea', 'latam', 'latin america']
        
        if any(region in segment.lower() for region in regions):
            return 'region'
        elif any(unit in segment.lower() for unit in ['cloud', 'software', 'hardware', 'services']):
            return 'business_unit'
        elif any(customer in segment.lower() for customer in ['consumer', 'enterprise', 'government']):
            return 'customer_type'
        else:
            return 'segment'
    
    def _associate_values_with_entities(self, text, entities, monetary_amounts, time_periods, metadata):
        """Associate monetary values with financial entities and time periods"""
        # If no entities found but we have monetary amounts, try to infer from context
        if not entities and monetary_amounts:
            context_lower = text.lower()
            # Try to infer the entity type from context
            for metric_type, metric_info in self.financial_metrics.items():
                for pattern in metric_info['patterns']:
                    if re.search(pattern, context_lower):
                        entities.append({
                            'type': metric_type,
                            'subtype': 'unknown',
                            'text': 'inferred ' + metric_type,
                            'position': re.search(pattern, context_lower).start(),
                            'dimensions': self._extract_entity_dimensions(context_lower),
                            'growth_applicable': metric_info['growth_applicable']
                        })
                        break
                if entities:  # Stop after finding the first match
                    break
            
            # If still no entities, use a generic "financial" entity
            if not entities:
                entities.append({
                    'type': 'financial',
                    'subtype': 'unknown',
                    'text': 'financial metric',
                    'position': 0,
                    'dimensions': self._extract_entity_dimensions(text),
                    'growth_applicable': True
                })
        
        # For each entity, find the closest monetary amount and time period
        for entity in entities:
            entity_pos = entity['position']
            
            # Find closest monetary amount
            closest_amount = None
            min_distance_amount = float('inf')
            
            for amount in monetary_amounts:
                distance = abs(entity_pos - amount['position'])
                if distance < min_distance_amount:
                    min_distance_amount = distance
                    closest_amount = amount
            
            # Find closest time period
            closest_period = None
            min_distance_period = float('inf')
            
            for period in time_periods:
                distance = abs(entity_pos - period['position'])
                if distance < min_distance_period:
                    min_distance_period = distance
                    closest_period = period
            
            # If we found both, create a financial data entry
            if closest_amount and closest_period:
                # Calculate confidence based on proximity
                distance_penalty = min(1.0, (min_distance_amount + min_distance_period) / 500)
                confidence = max(0.1, 0.9 - distance_penalty)
                
                # Get source from metadata
                source = metadata.get('source', 'unknown')
                
                # Create the financial data entry
                self.financial_data.append({
                    'entity_type': entity['type'],
                    'entity_subtype': entity['subtype'],
                    'dimensions': entity['dimensions'],
                    'period': closest_period['normalized'],
                    'value': closest_amount['value'],
                    'unit': closest_amount['unit'],
                    'confidence': confidence,
                    'source': source,
                    'growth_applicable': entity['growth_applicable'],
                    'context': text[:150] + '...' if len(text) > 150 else text
                })
    
    def _analyze_financial_data(self):
        """Analyze extracted financial data by entity type"""
        if not self.financial_data:
            logger.debug("No financial data found")
            return None
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.financial_data)
        
        # Save raw extracted data
        df.to_csv(self.output_dir / "financial_data_raw.csv", index=False)
        
        # Process each entity type separately
        entity_results = {}
        
        for entity_type in df['entity_type'].unique():
            # Filter for this entity type
            entity_df = df[df['entity_type'] == entity_type]
            
            # Further split by subtype if needed
            for subtype in entity_df['entity_subtype'].unique():
                # Create a key that combines type and subtype
                key = f"{entity_type}_{subtype}"
                if subtype == 'unknown' or subtype == entity_type:
                    key = entity_type
                
                # Filter for this subtype
                subtype_df = entity_df[entity_df['entity_subtype'] == subtype]
                
                # Process the subtypes with dimensions (like segments)
                dimension_results = self._process_entity_dimensions(subtype_df)
                
                # Process the aggregate level (ignoring dimensions)
                aggregate_results = self._process_entity_data(subtype_df)
                
                # Store results
                entity_results[key] = {
                    'aggregate': aggregate_results,
                    'dimensions': dimension_results
                }
                
                # Save individual entity data
                aggregate_results.to_csv(self.output_dir / f"{key}_data.csv", index=False)
        
        # Save a summary of all entity types
        entity_summary = {
            'entity_type': [],
            'entity_subtype': [],
            'latest_period': [],
            'latest_value': [],
            'latest_unit': [],
            'growth_rate': [],
            'data_points': []
        }
        
        for key, results in entity_results.items():
            aggregate_df = results['aggregate']
            if not aggregate_df.empty and 'period' in aggregate_df.columns:
                # Sort by period to get latest
                try:
                    aggregate_df['period_dt'] = aggregate_df['period'].apply(self._period_to_sortable_date)
                    sorted_df = aggregate_df.sort_values('period_dt', ascending=False)
                except:
                    sorted_df = aggregate_df.sort_values('period', ascending=False)
                
                # Get entity type and subtype
                if '_' in key:
                    entity_type, entity_subtype = key.split('_', 1)
                else:
                    entity_type = key
                    entity_subtype = 'all'
                
                # Add to summary
                entity_summary['entity_type'].append(entity_type)
                entity_summary['entity_subtype'].append(entity_subtype)
                entity_summary['latest_period'].append(sorted_df['period'].iloc[0])
                entity_summary['latest_value'].append(sorted_df['value'].iloc[0])
                entity_summary['latest_unit'].append(sorted_df['unit'].iloc[0])
                
                # Add growth rate if available
                growth_rate = sorted_df['growth_rate'].iloc[0] if 'growth_rate' in sorted_df.columns else None
                entity_summary['growth_rate'].append(growth_rate)
                
                # Count data points
                entity_summary['data_points'].append(len(sorted_df))
        
        # Save summary
        summary_df = pd.DataFrame(entity_summary)
        summary_df.to_csv(self.output_dir / "financial_entities_summary.csv", index=False)
        
        logger.debug(f"Processed {len(df)} financial data points across {len(entity_results)} entity types")
        return {
            'entity_results': entity_results,
            'summary': summary_df
        }
    
    def _process_entity_dimensions(self, entity_df):
        """Process entity data broken down by dimensions"""
        dimension_results = {}
        
        # Check if we have dimension data
        if 'dimensions' not in entity_df.columns or entity_df['dimensions'].isna().all():
            return dimension_results
        
        # Expand the dimensions to separate rows for each dimension value
        rows = []
        for idx, row in entity_df.iterrows():
            dimensions = row['dimensions']
            if isinstance(dimensions, dict) and dimensions:
                for dim_type, dim_values in dimensions.items():
                    for dim_value in dim_values:
                        new_row = row.drop('dimensions').copy()
                        new_row['dimension_type'] = dim_type
                        new_row['dimension_value'] = dim_value
                        rows.append(new_row)
        
        if not rows:
            return dimension_results
            
        # Create DataFrame with dimension data
        dim_df = pd.DataFrame(rows)
        
        # Group by dimension type and value
        for dim_type in dim_df['dimension_type'].unique():
            dim_type_df = dim_df[dim_df['dimension_type'] == dim_type]
            
            # Group by dimension value
            for dim_value in dim_type_df['dimension_value'].unique():
                dim_value_df = dim_type_df[dim_type_df['dimension_value'] == dim_value]
                
                # Process this dimension value
                processed_df = self._process_entity_data(dim_value_df)
                
                # Store result
                dim_key = f"{dim_type}_{dim_value}"
                dimension_results[dim_key] = processed_df
                
                # Save to file
                processed_df.to_csv(self.output_dir / f"{entity_df['entity_type'].iloc[0]}_{entity_df['entity_subtype'].iloc[0]}_{dim_key}_data.csv", index=False)
        
        return dimension_results
    
    def _process_entity_data(self, entity_df):
        """Process data for a single entity type"""
        if entity_df.empty:
            return pd.DataFrame()
        
        # Filter by confidence 
        df_high_conf = entity_df[entity_df['confidence'] >= 0.5]
        
        if df_high_conf.empty:
            df_high_conf = entity_df  # Use all data if no high confidence data
        
        # Group by period and calculate weighted average based on confidence
        df_high_conf['weighted_value'] = df_high_conf['value'] * df_high_conf['confidence']
        grouped = df_high_conf.groupby('period').agg({
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
            grouped['period_dt'] = grouped['period'].apply(self._period_to_sortable_date)
            grouped = grouped.sort_values('period_dt')
            grouped.drop('period_dt', axis=1, inplace=True)
        except:
            # If conversion fails, sort alphanumerically
            grouped = grouped.sort_values('period')
        
        # Add entity type information
        grouped['entity_type'] = entity_df['entity_type'].iloc[0]
        grouped['entity_subtype'] = entity_df['entity_subtype'].iloc[0]
        
        # Check if growth calculation is applicable
        growth_applicable = entity_df['growth_applicable'].iloc[0]
        
        # Calculate growth rates if applicable
        if growth_applicable and len(grouped) > 1:
            grouped['prev_value'] = grouped['value'].shift(1)
            grouped['growth_rate'] = (grouped['value'] - grouped['prev_value']) / grouped['prev_value']
            
            # Calculate annualized growth based on period type
            self._calculate_annualized_growth(grouped)
        
        return grouped
    
    def _period_to_sortable_date(self, period):
        """Convert period string to a sortable date format"""
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
        
        # Default fallback
        return period
    
    def _calculate_annualized_growth(self, df):
        """Calculate annual growth rates from different period types"""
        if 'growth_rate' not in df.columns:
            return df
            
        # Check the period pattern to determine period type
        periods = df['period'].tolist()
        
        # Default to no annualization
        df['annualized_growth'] = df['growth_rate']
        
        # If we can determine it's quarterly data
        if all(re.search(r'Q[1-4]', period) for period in periods):
            # Quarterly data
            df['annualized_growth'] = (1 + df['growth_rate']) ** 4 - 1
        # If we can determine it's monthly data
        elif all(re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', period) for period in periods):
            # Monthly data
            df['annualized_growth'] = (1 + df['growth_rate']) ** 12 - 1
        
        return df
    
    def visualize_entity_data(self, entity_type, entity_subtype=None):
        """Create visualizations for a specific financial entity"""
        # Determine the filename based on entity type and subtype
        if entity_subtype:
            filename = f"{entity_type}_{entity_subtype}_data.csv"
        else:
            filename = f"{entity_type}_data.csv"
        
        file_path = self.output_dir / filename
        
        if not file_path.exists():
            logger.debug(f"No data file found for {entity_type} {entity_subtype or ''}")
            return
        
        # Load the data
        df = pd.read_csv(file_path)
        
        if df.empty:
            logger.debug(f"No data available for {entity_type} {entity_subtype or ''}")
            return
        
        # Create visualizations
        plt.figure(figsize=(14, 10))
        
        # Entity values over time
        plt.subplot(2, 1, 1)
        plt.bar(df['period'], df['value'], color='#3498db')
        
        # Set title based on entity type and subtype
        if entity_subtype and entity_subtype != 'unknown':
            plt.title(f'{entity_subtype.capitalize()} Over Time', fontsize=16)
        else:
            plt.title(f'{entity_type.capitalize()} Over Time', fontsize=16)
            
        plt.ylabel(f'Value ({df["unit"].iloc[0]})', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Growth rate if available
        if 'growth_rate' in df.columns:
            plt.subplot(2, 1, 2)
            growth_values = df['growth_rate'] * 100
            colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in growth_values]
            plt.bar(df['period'], growth_values, color=colors)
            plt.title('Growth Rate (%)', fontsize=16)
            plt.ylabel('Growth Rate (%)', fontsize=12)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the visualization
        output_file = f"{entity_type}_{entity_subtype or 'all'}_visualization.png"
        plt.savefig(self.output_dir / output_file)
        plt.close()
        
        logger.debug(f"Visualization saved to {output_file}")