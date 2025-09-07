import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from collections import defaultdict, Counter
import structlog

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
                import openai
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
            # Extract time-based filters from query
            time_filter = self._extract_time_filter(query)
            store_filter = self._extract_store_filter(query)
            amount_filter = self._extract_amount_filter(query)

            # Build base query
            stmt = select(Receipt, ReceiptItem).join(
                ReceiptItem, Receipt.id == ReceiptItem.receipt_id, isouter=True
            )

            # Apply filters based on query content
            if time_filter:
                stmt = stmt.where(Receipt.receipt_date >= time_filter)

            if store_filter:
                stmt = stmt.where(Receipt.store_name.ilike(f"%{store_filter}%"))

            if amount_filter:
                if amount_filter['operator'] == 'greater':
                    stmt = stmt.where(Receipt.total_amount > amount_filter['value'])
                elif amount_filter['operator'] == 'less':
                    stmt = stmt.where(Receipt.total_amount < amount_filter['value'])

            stmt = stmt.limit(100)  # Increased limit for better analysis

            result = await db.execute(stmt)
            rows = result.fetchall()

            data = []
            for receipt, item in rows:
                receipt_data = {
                    "receipt_id": receipt.id,
                    "store_name": receipt.store_name,
                    "date": receipt.receipt_date.isoformat() if receipt.receipt_date else None,
                    "total": float(receipt.total_amount),
                    "item_name": item.item_name if item else None,
                    "item_price": float(item.total_price) if item else None,
                    "category": item.category if item else None,
                    "quantity": float(item.quantity) if item else None,
                }
                data.append(receipt_data)

            return data

        except Exception as e:
            logger.error("Failed to fetch relevant data", error=str(e))
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
            # Basic metrics
            unique_receipts = set(item["receipt_id"] for item in data if item["receipt_id"])
            total_spent = sum(item["total"] for item in data if item["total"])

            # Store analysis
            store_spending = defaultdict(float)
            store_visits = defaultdict(int)
            for item in data:
                if item["store_name"] and item["receipt_id"]:
                    store_spending[item["store_name"]] += item["total"]

            # Count unique visits per store
            receipt_stores = {}
            for item in data:
                if item["receipt_id"] and item["store_name"]:
                    receipt_stores[item["receipt_id"]] = item["store_name"]

            for store in receipt_stores.values():
                store_visits[store] += 1

            # Item analysis
            item_categories = Counter()
            item_spending = defaultdict(float)
            for item in data:
                if item["item_name"] and item["item_price"]:
                    item_spending[item["item_name"]] += item["item_price"]
                if item["category"]:
                    item_categories[item["category"]] += 1

            # Time analysis
            date_spending = defaultdict(float)
            for item in data:
                if item["date"] and item["total"]:
                    date = item["date"][:10]  # Extract date part
                    date_spending[date] += item["total"]

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
                    "min": min(item["total"] for item in data if item["total"]) if data else 0,
                    "max": max(item["total"] for item in data if item["total"]) if data else 0,
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
            # Prepare context for AI
            context = {
                "query": query,
                "data_summary": {
                    "total_receipts": len(data),
                    "receipts": data[:5],  # Limit data size for API
                },
            }

            system_prompt = """You are a helpful assistant that analyzes food purchase data. 
            Based on the receipt data provided, answer the user's question in a natural, conversational way.
            Focus on being accurate and helpful. If you can't find specific information, say so politely.
            Format monetary amounts with dollar signs and be specific about dates and locations when available."""

            response = self.client.chat.completions.create(
                model=settings.ai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nData: {json.dumps(context, indent=2)}",
                    },
                ],
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error("AI response generation failed", error=str(e))
            return await self._generate_rule_based_response(query, analysis, data)

    async def _generate_rule_based_response(
        self, query: str, analysis: Dict[str, Any], data: List[Dict[str, Any]]
    ) -> str:
        """Enhanced rule-based response generation with intelligent pattern matching"""
        try:
            query_lower = query.lower()
            total_receipts = analysis.get("total_receipts", 0)
            total_spent = analysis.get("total_spent", 0.0)
            
            # No data case
            if total_receipts == 0:
                return "I couldn't find any receipts matching your criteria. You might want to upload some receipts first or adjust your search parameters."
            
            # Spending amount queries
            if any(phrase in query_lower for phrase in ['how much', 'total spent', 'spent', 'cost', 'money']):
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
                    return "I couldn't find store information in your receipts."
                
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
            
            # Item/product queries
            elif any(phrase in query_lower for phrase in ['what', 'item', 'product', 'buy', 'bought', 'purchase']):
                top_items = analysis.get('top_items', {})
                categories = analysis.get('categories', {})
                
                if not top_items and not categories:
                    return "I found your receipts but couldn't identify specific items. The receipt text might need better processing."
                
                response = ""
                
                if top_items:
                    response += f"**Top items you've purchased:**\n"
                    for item, amount in list(top_items.items())[:8]:
                        response += f"• {item}: ${amount:.2f}\n"
                    response += "\n"
                
                if categories:
                    response += f"**Item categories:**\n"
                    for category, count in categories.items():
                        response += f"• {category}: {count} item{'s' if count != 1 else ''}\n"
                
                return response.strip()
            
            # Time-based queries
            elif any(phrase in query_lower for phrase in ['when', 'date', 'time', 'recent', 'last', 'this week', 'month']):
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
            
            # Comparison queries
            elif any(phrase in query_lower for phrase in ['compare', 'vs', 'versus', 'difference', 'more', 'less']):
                spending_range = analysis.get('spending_range', {})
                top_stores = analysis.get('top_stores', {})
                
                response = f"**Spending comparison insights:**\n\n"
                
                if spending_range:
                    response += f"• Lowest receipt: ${spending_range.get('min', 0):.2f}\n"
                    response += f"• Highest receipt: ${spending_range.get('max', 0):.2f}\n"
                    response += f"• Average per receipt: ${analysis.get('average_per_receipt', 0):.2f}\n\n"
                
                if len(top_stores) >= 2:
                    stores_list = list(top_stores.items())
                    top_store = stores_list[0]
                    second_store = stores_list[1]
                    difference = top_store[1] - second_store[1]
                    response += f"You spend ${difference:.2f} more at **{top_store[0]}** (${top_store[1]:.2f}) than at **{second_store[0]}** (${second_store[1]:.2f})."
                
                return response
            
            # Summary/general queries
            elif any(phrase in query_lower for phrase in ['summary', 'overview', 'tell me', 'show me', 'analyze']):
                response = f"**📊 Your Spending Summary:**\n\n"
                response += f"• **Total spent**: ${total_spent:.2f}\n"
                response += f"• **Number of receipts**: {total_receipts}\n"
                response += f"• **Average per receipt**: ${analysis.get('average_per_receipt', 0):.2f}\n"
                response += f"• **Stores visited**: {analysis.get('unique_stores', 0)}\n\n"
                
                # Top store
                top_stores = analysis.get('top_stores', {})
                if top_stores:
                    top_store = list(top_stores.items())[0]
                    response += f"🏪 **Top store**: {top_store[0]} (${top_store[1]:.2f})\n"
                
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
