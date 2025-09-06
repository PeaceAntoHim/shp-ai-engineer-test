import math

def cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors
    Formula: (A·B) / (|A| × |B|)
    """
    # Check if vectors have same length
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have same length")
    
    # Calculate dot product (A·B)
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    
    # Calculate magnitudes |A| and |B|
    magnitude_a = math.sqrt(sum(a * a for a in vector1))
    magnitude_b = math.sqrt(sum(b * b for b in vector2))
    
    # Avoid division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    # Return cosine similarity
    return dot_product / (magnitude_a * magnitude_b)


def find_most_similar(query_vector, vectors):
    """
    Find most similar vectors to query
    Returns list of (index, similarity_score) sorted by similarity
    """
    similarities = []
    
    for i, vector in enumerate(vectors):
        try:
            similarity = cosine_similarity(query_vector, vector)
            similarities.append((i, similarity))
        except:
            continue  # Skip invalid vectors
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities
