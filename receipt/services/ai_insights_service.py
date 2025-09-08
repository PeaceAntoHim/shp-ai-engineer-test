import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from collections import defaultdict, Counter
import structlog
import openai
from config.settings import get_settings
from schemas.insights_schema import InsightResponse, SpendingAnalytics, AIMode
from models.receipt_model import Receipt
from models.receipt_item_model import ReceiptItem

logger = structlog.get_logger()
settings = get_settings()


class AIService:
    """Service for AI-powered insights and analytics"""

    def __init__(self):
        self.client = None
        self._setup_ai_client()

    def _setup_ai_client(self):
        """Setup OpenAI client if API key is available"""
        try:
            if settings.openai_api_key:
                self.client = openai.OpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.warning("OpenAI API key not found, using rule-based responses")
        except Exception as e:
            logger.error("Failed to initialize OpenAI client", error=str(e))

    async def process_insight_query(
        self, question: str, db: AsyncSession, preferred_mode: Optional[AIMode] = None
    ) -> InsightResponse:
        """Process a natural language query about receipts"""
        try:
            # Fetch relevant data
            data = await self._fetch_relevant_data(question, db)

            # Analyze data for insights
            analysis = await self._analyze_data(question, data)

            # Determine AI mode to use
            ai_mode_used = self._determine_ai_mode(preferred_mode)

            # Generate response based on mode
            if ai_mode_used == AIMode.GENERATIVE and self.client:
                answer = await self._generate_ai_response(question, analysis, data)
                confidence = 0.8
            else:
                answer = await self._generate_rule_based_response(question, analysis, data)
                confidence = 0.7
                ai_mode_used = AIMode.RULE_BASED

            # Ensure all required fields are present and properly typed
            return InsightResponse(
                query=str(question),
                answer=str(answer),
                confidence=float(confidence),
                ai_mode_used=ai_mode_used,
                relevant_data=data[:5] if data else [],  # Limit to 5 items
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error("Failed to process insight query", error=str(e))
            # Return a safe default response
            return InsightResponse(
                query=str(question),
                answer="I apologize, but I encountered an error processing your question. Please try again.",
                confidence=0.0,
                ai_mode_used=AIMode.RULE_BASED,
                relevant_data=[],
                timestamp=datetime.now()
            )

    def _determine_ai_mode(self, preferred_mode: Optional[AIMode]) -> AIMode:
        """Determine which AI mode to use based on preference and availability"""
        if preferred_mode == AIMode.GENERATIVE and self.client:
            return AIMode.GENERATIVE
        elif preferred_mode == AIMode.RULE_BASED:
            return AIMode.RULE_BASED
        elif self.client:
            return AIMode.GENERATIVE
        else:
            return AIMode.RULE_BASED

    async def _fetch_relevant_data(
            self, query: str, db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Fetch relevant receipt data based on query with smart filtering"""
        try:
            logger.info(f"Fetching data for query: {query}")

            # First, let's check if we have any receipts at all with a simple query
            try:
                simple_stmt = select(Receipt.id, Receipt.store_name, Receipt.total_amount).limit(5)
                simple_result = await db.execute(simple_stmt)
                simple_rows = simple_result.fetchall()
                logger.info(f"Simple receipt query returned {len(simple_rows)} rows")
                if simple_rows:
                    logger.info(f"Sample receipt data: {simple_rows[0]}")
            except Exception as simple_error:
                logger.error(f"Simple receipt query failed: {str(simple_error)}")
                return []

            # Check total count
            try:
                count_stmt = select(func.count(Receipt.id))
                count_result = await db.execute(count_stmt)
                total_receipts = count_result.scalar()
                logger.info(f"Total receipts in database: {total_receipts}")

                if total_receipts == 0:
                    logger.warning("No receipts found in database")
                    return []
            except Exception as count_error:
                logger.error(f"Count query failed: {str(count_error)}")
                # Continue anyway, maybe the count is the problem

            # Extract filters from query
            time_filter = self._extract_time_filter(query)
            store_filter = self._extract_store_filter(query)
            amount_filter = self._extract_amount_filter(query)

            logger.info(f"Filters - time: {time_filter}, store: {store_filter}, amount: {amount_filter}")

            # FOR ITEM AND COMPARISON QUERIES, we need ALL receipt data
            query_lower = query.lower()
            needs_all_data = any(phrase in query_lower for phrase in [
                'compare', 'comparison', 'different stores', 'vs', 'versus',
                'what', 'item', 'items', 'buy', 'bought', 'frequently', 'frequent', 'purchase'
            ])

            logger.info(f"Needs all data (no store filtering): {needs_all_data}")

            # Try the simplest possible approach first - just get receipts
            try:
                logger.info("Trying simple receipts-only query")
                receipt_stmt = select(Receipt)

                # Only apply filters if not a broad query
                if not needs_all_data:
                    if time_filter:
                        receipt_stmt = receipt_stmt.where(Receipt.receipt_date >= time_filter)
                    if store_filter:
                        receipt_stmt = receipt_stmt.where(Receipt.store_name.ilike(f"%{store_filter}%"))
                    if amount_filter:
                        if amount_filter['operator'] == 'greater':
                            receipt_stmt = receipt_stmt.where(Receipt.total_amount > amount_filter['value'])
                        elif amount_filter['operator'] == 'less':
                            receipt_stmt = receipt_stmt.where(Receipt.total_amount < amount_filter['value'])

                receipt_stmt = receipt_stmt.order_by(Receipt.receipt_date.desc()).limit(100)

                receipt_result = await db.execute(receipt_stmt)
                receipts = receipt_result.scalars().all()

                logger.info(f"Simple receipt query returned {len(receipts)} receipts")

                if not receipts:
                    logger.warning("No receipts found even with simple query")
                    return []

                # Convert receipts to data format
                data = []
                for receipt in receipts:
                    receipt_data = {
                        "receipt_id": receipt.id,
                        "store_name": receipt.store_name,
                        "date": receipt.receipt_date.isoformat() if receipt.receipt_date else None,
                        "total": float(receipt.total_amount),
                        "item_name": None,
                        "item_price": None,
                        "category": None,
                        "quantity": None,
                    }
                    data.append(receipt_data)

                # Now try to get items for these receipts
                if receipts and any(phrase in query_lower for phrase in
                                    ['item', 'items', 'buy', 'bought', 'frequently', 'frequent', 'purchase', 'what']):
                    try:
                        logger.info("Fetching items for receipts")
                        receipt_ids = [r.id for r in receipts]
                        items_stmt = select(ReceiptItem).where(ReceiptItem.receipt_id.in_(receipt_ids)).limit(1000)
                        items_result = await db.execute(items_stmt)
                        items = items_result.scalars().all()

                        logger.info(f"Found {len(items)} items for these receipts")

                        # Create a lookup for receipt data
                        receipt_lookup = {r.id: r for r in receipts}

                        # Add item data
                        for item in items:
                            receipt = receipt_lookup.get(item.receipt_id)
                            if receipt:
                                item_data = {
                                    "receipt_id": item.receipt_id,
                                    "store_name": receipt.store_name,
                                    "date": receipt.receipt_date.isoformat() if receipt.receipt_date else None,
                                    "total": float(receipt.total_amount),
                                    "item_name": item.item_name,
                                    "item_price": float(item.total_price) if item.total_price else None,
                                    "category": item.category,
                                    "quantity": float(item.quantity) if item.quantity else None,
                                }
                                data.append(item_data)

                    except Exception as items_error:
                        logger.error(f"Failed to fetch items: {str(items_error)}")
                        # Continue with receipt data only

                logger.info(f"Final data length: {len(data)}")
                return data

            except Exception as receipt_error:
                logger.error(f"Simple receipt query failed: {str(receipt_error)}")
                return []

        except Exception as e:
            logger.error(f"Failed to fetch relevant data: {str(e)}", exc_info=True)
            return []
    def _extract_time_filter(self, query: str) -> Optional[datetime]:
        """Extract time-based filters from natural language query"""
        query_lower = query.lower()
        now = datetime.now()

        if any(phrase in query_lower for phrase in ['last week', 'past week', 'this week']):
            return now - timedelta(days=7)
        elif any(phrase in query_lower for phrase in ['last month', 'past month', 'this month']):
            return now - timedelta(days=30)
        elif any(phrase in query_lower for phrase in ['last 3 months', 'past 3 months']):
            return now - timedelta(days=90)
        elif any(phrase in query_lower for phrase in ['last year', 'past year', 'this year']):
            return now - timedelta(days=365)
        elif 'yesterday' in query_lower:
            return now - timedelta(days=1)
        elif 'today' in query_lower:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Extract number of days (e.g., "last 10 days")
        days_match = re.search(r'(?:last|past)\s+(\d+)\s+days?', query_lower)
        if days_match:
            days = int(days_match.group(1))
            return now - timedelta(days=days)

        return None

    def _extract_store_filter(self, query: str) -> Optional[str]:
        """Extract store names from query"""
        # Common store patterns
        store_patterns = [
            r'at\s+([a-zA-Z\s&]+)',
            r'from\s+([a-zA-Z\s&]+)',
            r'in\s+([a-zA-Z\s&]+)',
        ]

        for pattern in store_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                store_name = match.group(1).strip()
                if len(store_name) > 2:  # Filter out short matches
                    return store_name

        return None

    def _extract_amount_filter(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract amount-based filters from query"""
        query_lower = query.lower()

        # Pattern for amounts like "over $50", "more than $100", "less than $20"
        amount_patterns = [
            (r'over\s+\$?(\d+(?:\.\d{2})?)', 'greater'),
            (r'more than\s+\$?(\d+(?:\.\d{2})?)', 'greater'),
            (r'above\s+\$?(\d+(?:\.\d{2})?)', 'greater'),
            (r'less than\s+\$?(\d+(?:\.\d{2})?)', 'less'),
            (r'under\s+\$?(\d+(?:\.\d{2})?)', 'less'),
            (r'below\s+\$?(\d+(?:\.\d{2})?)', 'less'),
        ]

        for pattern, operator in amount_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return {
                    'operator': operator,
                    'value': float(match.group(1))
                }

        return None

    async def _analyze_data(
        self, query: str, data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enhanced data analysis with more insights"""
        if not data:
            return {"total_receipts": 0, "total_spent": 0.0}

        try:
            unique_receipts = set(item["receipt_id"] for item in data if item["receipt_id"])
            receipt_totals = {}
            for item in data:
                if item["receipt_id"] and item["total"]:
                    receipt_totals[item["receipt_id"]] = item["total"]
            total_spent = sum(receipt_totals.values())

            store_spending = defaultdict(float)
            store_visits = defaultdict(int)
            receipt_store_totals = {}
            for item in data:
                if item["store_name"] and item["receipt_id"] and item["total"]:
                    receipt_store_totals[item["receipt_id"]] = (item["store_name"], item["total"])
        
            for receipt_id, (store_name, total) in receipt_store_totals.items():
                store_spending[store_name] += total

            # Count unique visits per store
            receipt_stores = {}
            for item in data:
                if item["receipt_id"] and item["store_name"]:
                    receipt_stores[item["receipt_id"]] = item["store_name"]

            for store in receipt_stores.values():
                store_visits[store] += 1

            item_categories = Counter()
            item_spending = defaultdict(float)
            for item in data:
                if item["item_name"] and item["item_price"]:
                    item_spending[item["item_name"]] += item["item_price"]
                if item["category"]:
                    item_categories[item["category"]] += 1

            # Time analysis
            date_spending = defaultdict(float)
        
            # FIX: Only count each receipt once per date
            receipt_date_totals = {}
            for item in data:
                if item["date"] and item["receipt_id"] and item["total"]:
                    date = item["date"][:10]  # Extract date part
                    receipt_date_totals[item["receipt_id"]] = (date, item["total"])
        
            for receipt_id, (date, total) in receipt_date_totals.items():
                date_spending[date] += total

            analysis = {
                "total_receipts": len(unique_receipts),
                "total_spent": total_spent,
                "unique_stores": len(store_spending),
                "date_range": self._get_date_range(data),
                "top_stores": dict(sorted(store_spending.items(), key=lambda x: x[1], reverse=True)[:5]),
                "store_visits": dict(store_visits),
                "top_items": dict(sorted(item_spending.items(), key=lambda x: x[1], reverse=True)[:10]),
                "categories": dict(item_categories.most_common(5)),
                "daily_spending": dict(sorted(date_spending.items())),
                "average_per_receipt": total_spent / len(unique_receipts) if unique_receipts else 0,
                "spending_range": {
                    "min": min(receipt_totals.values()) if receipt_totals else 0,
                    "max": max(receipt_totals.values()) if receipt_totals else 0,
                }
            }

            return analysis

        except Exception as e:
            logger.error("Failed to analyze data", error=str(e))
            return {"total_receipts": 0, "total_spent": 0.0}

    def _get_date_range(self, data: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        """Get date range from data"""
        try:
            dates = [item["date"] for item in data if item["date"]]
            if not dates:
                return {"start": None, "end": None}

            return {
                "start": min(dates),
                "end": max(dates)
            }
        except Exception:
            return {"start": None, "end": None}

    async def _generate_ai_response(
            self, query: str, analysis: Dict[str, Any], data: List[Dict[str, Any]]
    ) -> str:
        """Generate AI response using OpenAI"""
        try:
            # Add debug information
            total_receipts = analysis.get("total_receipts", 0)
            logger.info(f"Generating AI response for query: {query}")
            logger.info(f"Analysis data: receipts={total_receipts}, stores={len(analysis.get('top_stores', {}))}")
            logger.info(f"Raw data length: {len(data)}")

            # Prepare context for AI using the corrected analysis data, not raw data
            context = {
                "query": query,
                "analysis": {
                    "total_receipts": analysis.get("total_receipts", 0),
                    "total_spent": analysis.get("total_spent", 0.0),
                    "average_per_receipt": analysis.get("average_per_receipt", 0.0),
                    "date_range": analysis.get("date_range", {}),
                    "top_stores": analysis.get("top_stores", {}),
                    "store_visits": analysis.get("store_visits", {}),
                    "top_items": analysis.get("top_items", {}),
                    "categories": analysis.get("categories", {}),
                    "daily_spending": analysis.get("daily_spending", {}),
                    "spending_range": analysis.get("spending_range", {})
                },
                "sample_receipts": self._get_unique_receipt_sample(data)
            }

            # Improve system prompt to handle limited data scenarios better
            system_prompt = """You are a helpful assistant that analyzes food purchase data. 
            Based on the analysis data provided, answer the user's question in a natural, conversational way.
            Use the pre-calculated totals and metrics from the analysis section - do NOT recalculate totals yourself.
            The analysis has already deduplicated receipt data, so use those values directly.

            If the user asks for comparisons but there's limited data (e.g., only one store), explain what you can see 
            and suggest what additional data would be needed for a proper comparison.

            If there's no data at all, politely explain that no receipt data was found and suggest uploading receipts.
            Format monetary amounts with dollar signs and be specific about dates and locations when available."""

            response = self.client.chat.completions.create(
                model=settings.ai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nAnalysis Data: {json.dumps(context, indent=2, default=str)}",
                    },
                ],
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error("AI response generation failed", error=str(e))
            return await self._generate_rule_based_response(query, analysis, data)

    def _get_unique_receipt_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get a sample of unique receipts to avoid sending duplicate data to AI"""
        unique_receipts = {}
        for item in data:
            receipt_id = item.get("receipt_id")
            if receipt_id and receipt_id not in unique_receipts:
                unique_receipts[receipt_id] = {
                    "receipt_id": receipt_id,
                    "store_name": item.get("store_name"),
                    "date": item.get("date"),
                    "total": item.get("total")
                }

        # Return up to 10 unique receipts as sample
        return list(unique_receipts.values())[:10]

    async def _generate_rule_based_response(
            self, query: str, analysis: Dict[str, Any], data: List[Dict[str, Any]]
    ) -> str:
        """Enhanced rule-based response generation with intelligent pattern matching"""
        try:
            query_lower = query.lower()
            total_receipts = analysis.get("total_receipts", 0)
            total_spent = analysis.get("total_spent", 0.0)

            # Debug logging
            logger.info(f"Rule-based response: query='{query}', receipts={total_receipts}, spent=${total_spent}")

            # No data case - but provide more specific guidance
            if total_receipts == 0:
                logger.info("No receipts found for rule-based response")
                if any(phrase in query_lower for phrase in
                       ['compare', 'comparison', 'different stores', 'vs', 'versus']):
                    return "I couldn't find receipts from multiple stores to compare. You might need to upload receipts from different stores first, or there might be only one store in your data."
                else:
                    return "I couldn't find any receipts matching your criteria. You might want to upload some receipts first or adjust your search parameters."

            # Item/product queries - HANDLE THIS CASE BETTER
            if any(phrase in query_lower for phrase in
                   ['what', 'item', 'product', 'buy', 'bought', 'purchase', 'frequently', 'frequent']):
                top_items = analysis.get('top_items', {})
                categories = analysis.get('categories', {})

                logger.info(f"Item query: found {len(top_items)} items, {len(categories)} categories")

                # If we have receipts but no items, explain the situation
                if not top_items and not categories:
                    return f"I found {total_receipts} receipt{'s' if total_receipts != 1 else ''} with a total of ${total_spent:.2f}, but I couldn't identify specific items from the receipt data. This might be because the receipt text processing needs improvement, or the receipts don't have detailed item information."

                response = ""

                if top_items:
                    response += f"**Items you buy frequently:**\n"
                    for item, amount in list(top_items.items())[:10]:
                        response += f"• {item}: ${amount:.2f}\n"
                    response += "\n"

                if categories:
                    response += f"**Item categories:**\n"
                    for category, count in categories.items():
                        response += f"• {category}: {count} item{'s' if count != 1 else ''}\n"
                    response += "\n"

                # Add summary information
                if response:
                    response += f"Based on {total_receipts} receipt{'s' if total_receipts != 1 else ''} totaling ${total_spent:.2f}."

                return response.strip()

            # Spending amount queries
            elif any(phrase in query_lower for phrase in ['how much', 'total spent', 'spent', 'cost', 'money']):
                response = f"Based on {total_receipts} receipt{'s' if total_receipts != 1 else ''}, you've spent **${total_spent:.2f}**"

                if 'average' in query_lower:
                    avg = analysis.get('average_per_receipt', 0)
                    response += f" with an average of ${avg:.2f} per receipt"

                # Add time context if available
                date_range = analysis.get('date_range', {})
                if date_range.get('start') and date_range.get('end'):
                    start_date = date_range['start'][:10]
                    end_date = date_range['end'][:10]
                    if start_date != end_date:
                        response += f" from {start_date} to {end_date}"
                    else:
                        response += f" on {start_date}"

                return response + "."

            # Store-related queries
            elif any(phrase in query_lower for phrase in ['where', 'store', 'shop', 'place', 'location']):
                top_stores = analysis.get('top_stores', {})
                store_visits = analysis.get('store_visits', {})

                if not top_stores:
                    return f"I found {total_receipts} receipt{'s' if total_receipts != 1 else ''} but couldn't identify store information. The receipts might need better store name extraction."

                response = f"You've shopped at **{len(top_stores)} different store{'s' if len(top_stores) != 1 else ''}**:\n\n"

                for store, amount in list(top_stores.items())[:5]:
                    visits = store_visits.get(store, 1)
                    avg_per_visit = amount / visits if visits > 0 else amount
                    response += f"• **{store}**: ${amount:.2f} across {visits} visit{'s' if visits != 1 else ''} (avg: ${avg_per_visit:.2f})\n"

                # Find favorite store
                if top_stores:
                    favorite_store = max(top_stores.items(), key=lambda x: x[1])
                    response += f"\nYour top spending location is **{favorite_store[0]}** at ${favorite_store[1]:.2f}."

                return response

            # Time-based queries
            elif any(phrase in query_lower for phrase in
                     ['when', 'date', 'time', 'recent', 'last', 'this week', 'month']):
                daily_spending = analysis.get('daily_spending', {})

                if not daily_spending:
                    return f"I found {total_receipts} receipts with a total of ${total_spent:.2f}, but couldn't determine the specific dates."

                response = f"**Your spending timeline:**\n\n"

                # Show recent spending
                sorted_dates = sorted(daily_spending.items(), reverse=True)
                for date, amount in sorted_dates[:10]:
                    formatted_date = datetime.fromisoformat(date).strftime("%B %d, %Y")
                    response += f"• {formatted_date}: ${amount:.2f}\n"

                # Find spending patterns
                if len(daily_spending) >= 7:
                    recent_week = sum(list(daily_spending.values())[-7:])
                    response += f"\nYour spending in the last 7 days: ${recent_week:.2f}"

                return response

            # Comparison queries - IMPROVED HANDLING
            elif any(phrase in query_lower for phrase in ['compare', 'vs', 'versus', 'difference', 'more', 'less']):
                spending_range = analysis.get('spending_range', {})
                top_stores = analysis.get('top_stores', {})

                # Handle store comparison specifically
                if any(phrase in query_lower for phrase in ['store', 'stores', 'shop', 'shops']):
                    if len(top_stores) < 2:
                        if len(top_stores) == 1:
                            store_name = list(top_stores.keys())[0]
                            return f"I found receipts from only one store: **{store_name}** (${list(top_stores.values())[0]:.2f}). To compare stores, you need receipts from multiple different stores."
                        else:
                            return f"I found {total_receipts} receipt{'s' if total_receipts != 1 else ''} but couldn't identify store information for comparison. The receipts might need better store name extraction."

                response = f"**Spending comparison insights:**\n\n"

                if spending_range:
                    response += f"• Lowest receipt: ${spending_range.get('min', 0):.2f}\n"
                    response += f"• Highest receipt: ${spending_range.get('max', 0):.2f}\n"
                    response += f"• Average per receipt: ${analysis.get('average_per_receipt', 0):.2f}\n\n"

                if len(top_stores) >= 2:
                    response += f"**Store comparison:**\n"
                    stores_list = list(top_stores.items())
                    for i, (store, amount) in enumerate(stores_list[:5]):
                        visits = analysis.get('store_visits', {}).get(store, 1)
                        avg_per_visit = amount / visits if visits > 0 else amount
                        response += f"{i + 1}. **{store}**: ${amount:.2f} ({visits} visit{'s' if visits != 1 else ''}, avg: ${avg_per_visit:.2f})\n"

                    # Show difference between top stores
                    if len(stores_list) >= 2:
                        top_store = stores_list[0]
                        second_store = stores_list[1]
                        difference = top_store[1] - second_store[1]
                        response += f"\nYou spend **${difference:.2f} more** at {top_store[0]} than at {second_store[0]}."
                elif len(top_stores) == 1:
                    store_name, amount = list(top_stores.items())[0]
                    response += f"I can only find data from one store: **{store_name}** (${amount:.2f}). Add receipts from other stores to enable comparison."

                return response

            # Summary/general queries
            elif any(phrase in query_lower for phrase in ['summary', 'overview', 'tell me', 'show me', 'analyze']):
                response = f"**Your Spending Summary:**\n\n"
                response += f"• **Total spent**: ${total_spent:.2f}\n"
                response += f"• **Number of receipts**: {total_receipts}\n"
                response += f"• **Average per receipt**: ${analysis.get('average_per_receipt', 0):.2f}\n"
                response += f"• **Stores visited**: {analysis.get('unique_stores', 0)}\n\n"

                # Top store
                top_stores = analysis.get('top_stores', {})
                if top_stores:
                    top_store = list(top_stores.items())[0]
                    response += f"**Top store**: {top_store[0]} (${top_store[1]:.2f})\n"

                # Date range
                date_range = analysis.get('date_range', {})
                if date_range.get('start') and date_range.get('end'):
                    start_date = date_range['start'][:10]
                    end_date = date_range['end'][:10]
                    response += f"**Period**: {start_date} to {end_date}\n"

                return response

            # Fallback with helpful suggestions
            else:
                suggestions = [
                    "• How much have I spent?",
                    "• Where do I shop the most?",
                    "• What items do I buy frequently?",
                    "• Show me my spending summary",
                    "• Compare my spending at different stores"
                ]

                response = f"I found {total_receipts} receipts with ${total_spent:.2f} in total spending, but I'm not sure exactly what you're looking for.\n\n"
                response += "Here are some questions you can ask:\n" + "\n".join(suggestions)

                return response

        except Exception as e:
            logger.error("Rule-based response generation failed", error=str(e))
            return "I'm sorry, I encountered an error while analyzing your receipts. Please try rephrasing your question."

    async def get_spending_analytics(
        self, db: AsyncSession, days: int = 30
    ) -> SpendingAnalytics:
        """Generate spending analytics for the specified period"""
        try:
            # Get all receipts in the database (ignoring the days filter for now)
            stmt = select(Receipt)
            result = await db.execute(stmt)
            receipts = result.scalars().all()
            
            if not receipts:
                return SpendingAnalytics(
                    total_spent=0.0,
                    transaction_count=0,
                    average_transaction=0.0,
                    top_categories=[],
                    daily_spending=[],
                    period_days=days
                )
            
            # Calculate metrics
            total_spent = sum(receipt.total_amount for receipt in receipts)
            transaction_count = len(receipts)
            average_transaction = total_spent / transaction_count if transaction_count > 0 else 0.0
            
            # Get receipt items for category analysis
            receipt_ids = [receipt.id for receipt in receipts]
            if receipt_ids:
                items_stmt = select(ReceiptItem).where(ReceiptItem.receipt_id.in_(receipt_ids))
                items_result = await db.execute(items_stmt)
                items = items_result.scalars().all()
                
                # Calculate top categories
                category_spending = defaultdict(float)
                for item in items:
                    if item.category:
                        category_spending[item.category] += float(item.total_price)
                
                top_categories = [
                    {"category": category, "amount": amount}
                    for category, amount in sorted(
                        category_spending.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                ]
            else:
                top_categories = []
            
            # Generate daily spending
            daily_spending = self._calculate_daily_spending(receipts, days)
            
            return SpendingAnalytics(
                total_spent=float(total_spent),
                transaction_count=int(transaction_count),
                average_transaction=float(average_transaction),
                top_categories=top_categories,
                daily_spending=daily_spending,
                period_days=int(days)
            )
            
        except Exception as e:
            logger.error("Failed to generate spending analytics", error=str(e))
            return SpendingAnalytics(
                total_spent=0.0,
                transaction_count=0,
                average_transaction=0.0,
                top_categories=[],
                daily_spending=[],
                period_days=days
            )
    
    def _calculate_daily_spending(self, receipts: List, days: int) -> List[Dict[str, Any]]:
        """Calculate daily spending breakdown"""
        try:
            daily_totals = {}
            
            for receipt in receipts:
                date_str = receipt.receipt_date.strftime("%Y-%m-%d")
                daily_totals[date_str] = daily_totals.get(date_str, 0.0) + float(receipt.total_amount)
            
            # Convert to list format
            daily_spending = [
                {"date": date, "amount": amount}
                for date, amount in sorted(daily_totals.items())
            ]
            
            return daily_spending
            
        except Exception as e:
            logger.error("Failed to calculate daily spending", error=str(e))
            return []
