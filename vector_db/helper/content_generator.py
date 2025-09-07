import random
import math

class ContentGenerator:
    """Generate dynamic content for vector database testing"""
    
    def __init__(self):
        # Base templates for each category
        self.animals_content = [
            "cats", "dogs", "birds", "fish", "rabbits", "hamsters", "horses", "cows", "sheep", "goats",
            "lions", "tigers", "elephants", "giraffes", "zebras", "pandas", "koalas", "kangaroos",
            "domestic pets", "wild animals", "farm animals", "zoo animals", "pet care", "veterinary",
            "animal behavior", "wildlife conservation", "pet training", "animal nutrition"
        ]
        
        self.programming_content = [
            "Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "Ruby", "PHP", "Swift",
            "web development", "mobile apps", "desktop applications", "databases", "algorithms",
            "data structures", "machine learning", "artificial intelligence", "receipt development",
            "frontend development", "full stack", "DevOps", "cloud computing", "microservices"
        ]
        
        self.ai_content = [
            "machine learning", "deep learning", "neural networks", "natural language processing",
            "computer vision", "reinforcement learning", "supervised learning", "unsupervised learning",
            "artificial intelligence", "data science", "big data", "predictive analytics",
            "recommendation systems", "chatbots", "automation", "robotics", "AI ethics"
        ]
        
        self.business_content = [
            "marketing", "sales", "finance", "accounting", "human resources", "management",
            "strategy", "operations", "customer service", "business development", "entrepreneurship",
            "startups", "consulting", "project management", "leadership", "teamwork", "productivity"
        ]
        
        # Content patterns
        self.content_patterns = [
            "Learn about {topic} and improve your skills",
            "Understanding {topic} for beginners",
            "Advanced {topic} techniques and strategies", 
            "Complete guide to {topic}",
            "Best practices for {topic}",
            "{topic} tutorial and examples",
            "How to master {topic} effectively",
            "Professional {topic} development",
            "Essential {topic} concepts",
            "{topic} tips and tricks"
        ]
    
    def generate_animal_content(self, count=10):
        """Generate animal-related content"""
        content_list = []
        
        for i in range(count):
            # Pick random animal topic
            topic = random.choice(self.animals_content)
            pattern = random.choice(self.content_patterns)
            content = pattern.format(topic=topic)
            
            # Generate embedding (similar to animal base vector [1.0, 0.0, 0.0, 0.0])
            embedding = self._generate_similar_embedding([1.0, 0.0, 0.0, 0.0], variance=0.3)
            
            # Create metadata
            metadata = {
                "category": "animals",
                "type": topic,
                "generated": True,
                "index": i
            }
            
            content_list.append({
                "embedding": embedding,
                "content": content,
                "metadata": metadata
            })
        
        return content_list
    
    def generate_programming_content(self, count=10):
        """Generate programming-related content"""
        content_list = []
        
        for i in range(count):
            topic = random.choice(self.programming_content)
            pattern = random.choice(self.content_patterns)
            content = pattern.format(topic=topic)
            
            # Generate embedding (similar to programming base vector [0.0, 1.0, 0.0, 0.0])
            embedding = self._generate_similar_embedding([0.0, 1.0, 0.0, 0.0], variance=0.3)
            
            metadata = {
                "category": "programming",
                "language": topic if topic in ["Python", "JavaScript", "Java", "C++", "C#"] else "general",
                "generated": True,
                "index": i
            }
            
            content_list.append({
                "embedding": embedding,
                "content": content,
                "metadata": metadata
            })
        
        return content_list
    
    def generate_ai_content(self, count=10):
        """Generate AI-related content"""
        content_list = []
        
        for i in range(count):
            topic = random.choice(self.ai_content)
            pattern = random.choice(self.content_patterns)
            content = pattern.format(topic=topic)
            
            # Generate embedding (similar to AI base vector [0.0, 0.0, 1.0, 0.0])
            embedding = self._generate_similar_embedding([0.0, 0.0, 1.0, 0.0], variance=0.3)
            
            metadata = {
                "category": "ai",
                "topic": topic.replace(" ", "_"),
                "generated": True,
                "index": i
            }
            
            content_list.append({
                "embedding": embedding,
                "content": content,
                "metadata": metadata
            })
        
        return content_list
    
    def generate_business_content(self, count=10):
        """Generate business-related content"""
        content_list = []
        
        for i in range(count):
            topic = random.choice(self.business_content)
            pattern = random.choice(self.content_patterns)
            content = pattern.format(topic=topic)
            
            # Generate embedding (similar to business base vector [0.0, 0.0, 0.0, 1.0])
            embedding = self._generate_similar_embedding([0.0, 0.0, 0.0, 1.0], variance=0.3)
            
            metadata = {
                "category": "business",
                "field": topic,
                "generated": True,
                "index": i
            }
            
            content_list.append({
                "embedding": embedding,
                "content": content,
                "metadata": metadata
            })
        
        return content_list
    
    def generate_mixed_content(self, total_count=40):
        """Generate mixed content from all categories"""
        # Generate equal amounts from each category
        per_category = total_count // 4
        
        all_content = []
        all_content.extend(self.generate_animal_content(per_category))
        all_content.extend(self.generate_programming_content(per_category))
        all_content.extend(self.generate_ai_content(per_category))
        all_content.extend(self.generate_business_content(per_category))
        
        # Shuffle the list
        random.shuffle(all_content)
        
        return all_content
    
    def _generate_similar_embedding(self, base_vector, variance=0.2):
        """Generate embedding similar to base vector with some variance"""
        new_vector = []
        
        for value in base_vector:
            if value > 0:
                # Add some random variance to non-zero values
                noise = random.uniform(-variance, variance)
                new_value = max(0.0, min(1.0, value + noise))
            else:
                # Add small random noise to zero values
                new_value = random.uniform(0.0, variance/2)
            
            new_vector.append(new_value)
        
        # Normalize vector to maintain similar magnitude
        magnitude = math.sqrt(sum(x * x for x in new_vector))
        if magnitude > 0:
            new_vector = [x / magnitude for x in new_vector]
        
        return new_vector
    
    def create_query_vectors(self):
        """Create sample query vectors for testing"""
        queries = {
            "animal_query": {
                "vector": [0.9, 0.1, 0.0, 0.0],
                "description": "Looking for animal content"
            },
            "programming_query": {
                "vector": [0.0, 0.9, 0.1, 0.0],
                "description": "Looking for programming content"
            },
            "ai_query": {
                "vector": [0.0, 0.0, 0.9, 0.1],
                "description": "Looking for AI content"
            },
            "business_query": {
                "vector": [0.0, 0.0, 0.1, 0.9],
                "description": "Looking for business content"
            },
            "mixed_query": {
                "vector": [0.25, 0.25, 0.25, 0.25],
                "description": "Looking for mixed content"
            }
        }
        
        return queries


def demo_content_generation():
    """Demo the content generator"""
    generator = ContentGenerator()
    
    print("=== Dynamic Content Generator Demo ===\n")
    
    # Generate content for each category
    categories = ["animal", "programming", "ai", "business"]
    
    for category in categories:
        print(f"--- {category.upper()} CONTENT ---")
        
        if category == "animal":
            content = generator.generate_animal_content(3)
        elif category == "programming":
            content = generator.generate_programming_content(3)
        elif category == "ai":
            content = generator.generate_ai_content(3)
        else:
            content = generator.generate_business_content(3)
        
        for i, item in enumerate(content, 1):
            print(f"{i}. {item['content']}")
            print(f"   Embedding: {[f'{x:.2f}' for x in item['embedding']]}")
            print(f"   Metadata: {item['metadata']}")
            print()
    
    # Show query examples
    print("--- SAMPLE QUERY VECTORS ---")
    queries = generator.create_query_vectors()
    
    for name, query in queries.items():
        print(f"{name}: {query['vector']} - {query['description']}")
