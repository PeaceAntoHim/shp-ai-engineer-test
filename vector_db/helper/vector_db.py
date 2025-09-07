import time
from helper.connection import Database
from helper.cosine_similarity import cosine_similarity, find_most_similar
from helper.logger import VectorDBLogger


class VectorDB:
    """
    Simple Vector Database with comprehensive logging and metrics
    Easy to use and understand
    """

    def __init__(self, enable_logging=True):
        """Initialize vector database with optional logging"""
        print("Starting Vector Database...")
        self.db = Database()

        # Initialize logger
        self.logger = VectorDBLogger() if enable_logging else None

        if self.logger:
            self.logger.start_operation("VectorDB_Initialization")
            self.logger.end_operation()

    def add(self, embedding, content=None, metadata=None):
        """
        Add a vector to database with logging

        Args:
            embedding: List of numbers (vector)
            content: Text content (optional)
            metadata: Dictionary with extra info (optional)

        Returns:
            int: Vector ID
        """
        if not isinstance(embedding, list):
            raise ValueError("Embedding must be a list of numbers")

        if not all(isinstance(x, (int, float)) for x in embedding):
            raise ValueError("All embedding values must be numbers")

        start_time = time.time()

        vector_id = self.db.insert_vector(embedding, content, metadata)

        if self.logger:
            self.logger.log_vector_addition(vector_id, len(embedding))
            self.logger.log_database_operation("INSERT_VECTOR", True)

        print(f"Added vector {vector_id}")
        return vector_id

    def search(self, query_vector, limit=5, min_similarity=0.0):
        """
        Search for similar vectors with detailed logging

        Args:
            query_vector: List of numbers to search for
            limit: Maximum number of results
            min_similarity: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of dictionaries with vector info and similarity scores
        """
        search_start_time = time.time()

        # Get all vectors from database
        all_vectors = self.db.get_all_vectors()

        if not all_vectors:
            print("No vectors in database")
            return []

        # Extract just the embeddings for similarity calculation
        embeddings = [v["embedding"] for v in all_vectors]

        # Log similarity calculations
        similarity_start_time = time.time()

        # Find similar vectors
        similarities = find_most_similar(query_vector, embeddings)

        similarity_end_time = time.time()
        similarity_calculation_time = similarity_end_time - similarity_start_time

        # Prepare results
        results = []
        similarity_scores = []

        for index, similarity in similarities:
            if similarity >= min_similarity and len(results) < limit:
                vector_data = all_vectors[index].copy()
                vector_data["similarity"] = similarity
                results.append(vector_data)
                similarity_scores.append(similarity)

        search_end_time = time.time()
        total_search_time = search_end_time - search_start_time

        # Log search operation
        if self.logger:
            self.logger.log_search_operation(
                query_dimension=len(query_vector),
                results_count=len(results),
                search_time=total_search_time,
                similarities=similarity_scores,
            )
            self.logger.log_similarity_calculation(
                calculation_count=len(embeddings),
                calculation_time=similarity_calculation_time,
            )
            self.logger.log_database_operation("SELECT_VECTORS", True)

        print(f"Found {len(results)} similar vectors")
        return results

    def get(self, vector_id):
        """Get vector by ID with logging"""
        result = self.db.get_vector(vector_id)

        if self.logger:
            self.logger.log_database_operation(
                "SELECT_VECTOR_BY_ID", result is not None
            )

        return result

    def delete(self, vector_id):
        """Delete vector by ID with logging"""
        success = self.db.delete_vector(vector_id)

        if self.logger:
            self.logger.log_vector_deletion(vector_id, success)
            self.logger.log_database_operation("DELETE_VECTOR", success)

        return success

    def list_all(self):
        """List all vectors in database with logging"""
        vectors = self.db.get_all_vectors()

        if self.logger:
            self.logger.log_database_operation("SELECT_ALL_VECTORS", True)

        return vectors

    def calculate_similarity(self, vector1, vector2):
        """Calculate cosine similarity between two vectors with logging"""
        start_time = time.time()
        similarity = cosine_similarity(vector1, vector2)
        end_time = time.time()

        calculation_time = end_time - start_time

        if self.logger:
            self.logger.log_similarity_calculation(1, calculation_time)

        return similarity

    def get_statistics(self):
        """Get database statistics"""
        all_vectors = self.list_all()

        if not all_vectors:
            return {
                "total_vectors": 0,
                "average_dimension": 0,
                "categories": {},
                "dimensions_distribution": {},
            }

        # Calculate statistics
        total_vectors = len(all_vectors)
        dimensions = [len(v["embedding"]) for v in all_vectors]
        average_dimension = sum(dimensions) / len(dimensions) if dimensions else 0

        # Category distribution
        categories = {}
        for vector in all_vectors:
            metadata = vector.get("metadata", {})
            category = metadata.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1

        # Dimension distribution
        dimensions_dist = {}
        for dim in dimensions:
            dimensions_dist[dim] = dimensions_dist.get(dim, 0) + 1

        stats = {
            "total_vectors": total_vectors,
            "average_dimension": average_dimension,
            "categories": categories,
            "dimensions_distribution": dimensions_dist,
            "logger_metrics": self.logger.metrics.to_dict() if self.logger else None,
        }

        return stats

    def close(self, operation_type=""):
        """Close database connection and generate final report"""
        if self.logger:
            self.logger.start_operation("VectorDB_Cleanup")
            self.logger.end_operation()

        self.db.close()

        # Generate comprehensive report
        if self.logger:
            report_path = self.logger.generate_comprehensive_report(operation_type)
            print(f"Comprehensive analysis report generated: {report_path}")
            return report_path

        return None
