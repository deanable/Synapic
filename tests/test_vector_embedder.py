"""
Test Vector Embedder Functionality
===================================

Tests for the semantic taxonomy vectorization system.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Test the basic functionality even if dependencies are not available
def test_vector_embedder_import():
    """Test that vector embedder can be imported."""
    try:
        from src.core.vector_embedder import (  # noqa: F401
            VectorDatabase, TextEmbedder, SemanticTaxonomy,
            enhance_metadata_with_semantics, setup_vector_system
        )
        assert True  # Import successful
    except ImportError as e:
        # This is expected if dependencies are not installed
        print(f"Vector dependencies not available (expected in test): {e}")
        assert "sentence-transformers" in str(e) or "faiss" in str(e)


def test_dummy_taxonomy():
    """Test that dummy taxonomy works when dependencies are missing."""
    with patch('src.core.vector_embedder.HAS_VECTOR_DEPS', False):
        from src.core.vector_embedder import DummySemanticTaxonomy, get_semantic_taxonomy
        
        # Test dummy taxonomy
        dummy = DummySemanticTaxonomy()
        dummy.add_tags(["test", "sample"])
        similar = dummy.find_similar_tags("test")
        assert similar == []
        
        stats = dummy.get_taxonomy_stats()
        assert stats['tag_count'] == 0
        assert stats['has_vector_deps'] is False
        
        # Test get_semantic_taxonomy returns dummy when deps missing
        taxonomy = get_semantic_taxonomy()
        assert isinstance(taxonomy, DummySemanticTaxonomy)


def test_semantic_enhancement_without_deps():
    """Test semantic enhancement when dependencies are missing."""
    with patch('src.core.vector_embedder.HAS_VECTOR_DEPS', False):
        from src.core.vector_embedder import enhance_metadata_with_semantics, DummySemanticTaxonomy
        
        dummy_taxonomy = DummySemanticTaxonomy()
        
        result = enhance_metadata_with_semantics(
            category="Nature",
            keywords=["tree", "forest"],
            description="A beautiful forest scene",
            taxonomy=dummy_taxonomy
        )
        
        assert result['semantic_enabled'] is False
        assert 'message' in result


def test_vector_database_initialization():
    """Test vector database initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.faiss"
        
        with patch('src.core.vector_embedder.HAS_VECTOR_DEPS', True), \
             patch('src.core.vector_embedder.faiss') as mock_faiss:
            
            # Mock FAISS index
            mock_index = MagicMock()
            mock_index.ntotal = 0
            mock_faiss.IndexFlatL2.return_value = mock_index
            mock_faiss.read_index.side_effect = FileNotFoundError()
            
            from src.core.vector_embedder import VectorDatabase
            
            # Test new index creation
            db = VectorDatabase(db_path, dimension=384)
            assert db.index == mock_index
            assert db.get_embedding_count() == 0


def test_text_embedder_initialization():
    """Test text embedder initialization."""
    with patch('src.core.vector_embedder.HAS_VECTOR_DEPS', True), \
         patch('src.core.vector_embedder.SentenceTransformer') as mock_st:
        
        # Mock SentenceTransformer
        mock_model = MagicMock()
        # encode() returns a 2D array (n, dim); .shape[1] is read after loading
        encoded = MagicMock()
        encoded.shape = (1, 384)
        mock_model.encode.return_value = encoded
        mock_st.return_value = mock_model
        
        from src.core.vector_embedder import TextEmbedder
        
        embedder = TextEmbedder(model_name="test-model")
        assert embedder.model == mock_model
        assert embedder.model_name == "test-model"


def test_integration_with_image_processing():
    """Test that image processing can use semantic enhancement."""
    from src.core.image_processing import extract_tags_with_semantics
    
    # Mock result data
    mock_result = [
        {
            'label': 'nature, landscape',
            'score': 0.95
        },
        {
            'label': 'tree',
            'score': 0.87
        }
    ]
    
    # Test without taxonomy (should still work)
    category, keywords, description, semantic_data = extract_tags_with_semantics(
        result=mock_result,
        model_task="image-classification",
        taxonomy=None
    )
    
    assert category == ""  # No category in this result format
    assert "Nature" in keywords or "Landscape" in keywords  # Title-cased versions
    assert semantic_data['semantic_enabled'] is False


if __name__ == "__main__":
    print("Running vector embedder tests...")
    
    test_vector_embedder_import()
    print("✓ Import test passed")
    
    test_dummy_taxonomy()
    print("✓ Dummy taxonomy test passed")
    
    test_semantic_enhancement_without_deps()
    print("✓ Semantic enhancement test passed")
    
    test_vector_database_initialization()
    print("✓ Vector database test passed")
    
    test_text_embedder_initialization()
    print("✓ Text embedder test passed")
    
    test_integration_with_image_processing()
    print("✓ Image processing integration test passed")
    
    print("\nAll tests completed!")