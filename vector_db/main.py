import time
from helper.vector_db import VectorDB
from helper.content_generator import ContentGenerator


def main():
    print("=== Vector Database with Dynamic Content & Logging ===\n")

    # Create vector database with logging enabled
    vdb = VectorDB(enable_logging=True)
    generator = ContentGenerator()

    # Start main operation tracking
    if vdb.logger:
        vdb.logger.start_operation("Complete_Vector_Analysis")

    # Generate dynamic content (40 items total, 10 per category)
    print("1. Generating dynamic content...")
    all_content = generator.generate_mixed_content(40)
    print(f"Generated {len(all_content)} content items\n")

    # Add all content to database
    print("2. Adding content to vector database...")
    vector_ids = []

    add_start_time = time.time()
    for i, item in enumerate(all_content):
        vector_id = vdb.add(
            embedding=item["embedding"],
            content=item["content"],
            metadata=item["metadata"]
        )
        vector_ids.append(vector_id)

        # Show progress every 10 items
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i + 1}/{len(all_content)} vectors added")

    add_end_time = time.time()
    add_duration = add_end_time - add_start_time

    print(f"Added {len(vector_ids)} vectors in {add_duration:.4f} seconds")
    print(f"Average addition time: {add_duration / len(vector_ids):.6f} seconds per vector\n")

    # Test different types of queries
    print("3. Testing different query types...")
    queries = generator.create_query_vectors()

    search_times = []
    total_results_found = 0

    for query_name, query_info in queries.items():
        print(f"\n--- {query_name.upper().replace('_', ' ')} ---")
        print(f"Query: {query_info['description']}")
        print(f"Vector: {query_info['vector']}")

        # Search for similar content
        search_start = time.time()
        results = vdb.search(
            query_vector=query_info['vector'],
            limit=3,
            min_similarity=0.0  # Low threshold for demo
        )
        search_end = time.time()
        search_time = search_end - search_start
        search_times.append(search_time)
        total_results_found += len(results)

        if results:
            print("Top Results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Similarity: {result['similarity']:.3f}")
                print(f"     Content: {result['content'][:60]}...")
                print(f"     Category: {result['metadata']['category']}")
                print()
        else:
            print("No results found\n")

    # Performance summary
    avg_search_time = sum(search_times) / len(search_times) if search_times else 0
    print(f"Search Performance Summary:")
    print(f"   Total searches: {len(search_times)}")
    print(f"   Average search time: {avg_search_time:.4f} seconds")
    print(f"   Total results found: {total_results_found}")
    print(f"   Average results per search: {total_results_found / len(search_times) if search_times else 0:.1f}")

    # Category-specific analysis
    print("\n4. Analyzing content by category...")

    categories = ["animals", "programming", "ai", "business"]
    category_analysis = {}

    for category in categories:
        # Count vectors in this category
        all_vectors = vdb.list_all()
        category_count = sum(1 for v in all_vectors if v['metadata']['category'] == category)
        category_analysis[category] = category_count
        print(f"{category.capitalize()}: {category_count} items")

    print(f"\nTotal vectors in database: {len(all_vectors)}")

    # Test similarity between different categories
    print("\n5. Cross-category similarity analysis...")

    # Get sample vectors from each category
    samples = {}
    for vector in all_vectors[:20]:  # Check first 20 for efficiency
        category = vector['metadata']['category']
        if category not in samples:
            samples[category] = vector['embedding']

    # Calculate cross-category similarities
    print("Cross-category similarity matrix:")
    print("Category".ljust(12), end="")
    for cat in categories:
        if cat in samples:
            print(f"{cat[:8].capitalize()}".ljust(10), end="")
    print()

    similarity_matrix = []
    for cat1 in categories:
        if cat1 in samples:
            row = [cat1.capitalize().ljust(12)]
            for cat2 in categories:
                if cat2 in samples:
                    similarity = vdb.calculate_similarity(samples[cat1], samples[cat2])
                    row.append(f"{similarity:.3f}".ljust(10))
                    similarity_matrix.append(similarity)
                else:
                    row.append("N/A".ljust(10))
            print("".join(row))

    # Performance test with generated content
    print("\n6. Batch performance test...")

    # Generate test queries
    test_queries = [generator.create_query_vectors()["mixed_query"]["vector"] for _ in range(5)]

    batch_start_time = time.time()
    batch_results = []

    for i, query in enumerate(test_queries):
        results = vdb.search(query, limit=5)
        batch_results.append(results)
        print(f"   Batch query {i + 1}: {len(results)} results")

    batch_end_time = time.time()
    batch_duration = batch_end_time - batch_start_time

    total_batch_results = sum(len(r) for r in batch_results)

    print(f"Batch Performance:")
    print(f"   Total batch time: {batch_duration:.4f} seconds")
    print(f"   Average time per query: {batch_duration / len(test_queries):.4f} seconds")
    print(f"   Total results: {total_batch_results}")

    # Database statistics
    print("\n7. Database statistics...")
    stats = vdb.get_statistics()

    print(f"Database Statistics:")
    print(f"   Total vectors: {stats['total_vectors']:,}")
    print(f"   Average dimension: {stats['average_dimension']:.1f}")
    print(f"   Categories: {list(stats['categories'].keys())}")
    print(f"   Category distribution:")
    for category, count in stats['categories'].items():
        percentage = (count / stats['total_vectors']) * 100
        print(f"      {category}: {count} ({percentage:.1f}%)")

    # End main operation tracking
    if vdb.logger:
        vdb.logger.end_operation("full_vector_analysis")

    # Memory and performance summary
    print("\n8. Performance Summary...")
    if vdb.logger:
        metrics = vdb.logger.metrics
        print(f"Performance Metrics:")
        print(f"   Total execution time: {metrics.execution_time:.4f} seconds")
        print(f"   Memory usage: {metrics.memory_usage_mb:.1f} MB")
        print(f"   Peak memory: {metrics.peak_memory_mb:.1f} MB")
        print(f"   CPU usage: {metrics.cpu_usage_percent:.1f}%")
        print(f"   Vectors processed: {metrics.vectors_processed:,}")
        print(f"   Search queries: {metrics.search_queries:,}")
        print(f"   Similarity calculations: {metrics.total_similarity_calculations:,}")

    # # Cleanup and generate final report
    # print("\n9. Cleanup and report generation...")
    # deleted_count = 0
    # cleanup_start_time = time.time()
    #
    # for vector_id in vector_ids:
    #     if vdb.delete(vector_id):
    #         deleted_count += 1
    #
    # cleanup_end_time = time.time()
    # cleanup_duration = cleanup_end_time - cleanup_start_time
    #
    # print(f"Cleaned up {deleted_count}/{len(vector_ids)} vectors in {cleanup_duration:.4f} seconds")

    # Close database and generate comprehensive report
    print("\n10. Generating comprehensive analysis report...")
    report_path = vdb.close("full_vector_analysis")

    if report_path:
        print(f"Detailed analysis report available at: {report_path}")

    print("\nDemo completed successfully!")


def performance_focused_demo():
    """Performance-focused demo with detailed metrics"""
    print("\n=== Performance-Focused Vector DB Demo ===")

    vdb = VectorDB(enable_logging=True)
    generator = ContentGenerator()

    if vdb.logger:
        vdb.logger.start_operation("Performance_Analysis")

    # Test with different dataset sizes
    dataset_sizes = [10, 50, 100]

    for size in dataset_sizes:
        print(f"\n--- Testing with {size} vectors ---")

        # Generate content
        content = generator.generate_mixed_content(size)

        # Add vectors and measure time
        add_start = time.time()
        vector_ids = []

        for item in content:
            vector_id = vdb.add(
                embedding=item["embedding"],
                content=item["content"],
                metadata=item["metadata"]
            )
            vector_ids.append(vector_id)

        add_end = time.time()
        add_time = add_end - add_start

        # Test search performance
        query = generator.create_query_vectors()["mixed_query"]["vector"]

        search_start = time.time()
        results = vdb.search(query, limit=10)
        search_end = time.time()
        search_time = search_end - search_start

        # # Cleanup
        # for vid in vector_ids:
        #     vdb.delete(vid)

        print(f"Results for {size} vectors:")
        print(f"   Add time: {add_time:.4f}s ({add_time / size:.6f}s per vector)")
        print(f"   Search time: {search_time:.4f}s")
        print(f"   Results found: {len(results)}")

    if vdb.logger:
        vdb.logger.end_operation("performance_analysis")

    # Generate performance report
    report_path = vdb.close("performance_analysis")
    print(f"\nPerformance analysis completed. Report: {report_path}")


if __name__ == "__main__":
    main()
    performance_focused_demo()