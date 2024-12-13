import pytest
from llmeasy.utils.json_helper import JSONStreamHelper
import json
import logging
import asyncio
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

class TestJSONStreamHelper:
    @pytest.fixture
    def helper(self):
        return JSONStreamHelper()

    async def generate_chunks(self, chunks) -> AsyncGenerator[str, None]:
        """Helper to simulate a stream of chunks"""
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.01)  # Small delay to simulate real streaming

    @pytest.mark.asyncio
    async def test_basic_json_streaming(self, helper):
        """Test basic JSON streaming functionality"""
        chunks = [
            '{"key": "val',
            'ue", "num',
            'ber": 42}'
        ]
        
        objects = []
        async for obj in helper.process_stream(self.generate_chunks(chunks)):
            objects.append(obj)
            
        assert len(objects) == 1
        assert objects[0] == {"key": "value", "number": 42}

    @pytest.mark.asyncio
    async def test_multiple_objects(self, helper):
        """Test streaming multiple JSON objects"""
        chunks = [
            '{"first": true}',
            ' {"second": ',
            '42} {"third": "value"}'
        ]
        
        objects = []
        async for obj in helper.process_stream(self.generate_chunks(chunks)):
            objects.append(obj)
            
        assert len(objects) == 3
        assert objects[0] == {"first": True}
        assert objects[1] == {"second": 42}
        assert objects[2] == {"third": "value"}

    @pytest.mark.asyncio
    async def test_malformed_json(self, helper):
        """Test handling of malformed JSON"""
        chunks = [
            '{"first": true}',
            ' {"invalid": missing_quotes} ',
            '{"third": "value"}'
        ]
        
        objects = []
        async for obj in helper.process_stream(
            self.generate_chunks(chunks),
            repair_json=False
        ):
            objects.append(obj)
            
        assert len(objects) == 2  # Should get first and third objects
        assert objects[0] == {"first": True}
        assert objects[1] == {"third": "value"}

    @pytest.mark.asyncio
    async def test_nested_objects(self, helper):
        """Test handling of nested JSON objects"""
        chunks = [
            '{"outer": {"inn',
            'er": {"value": 42',
            '}}, "array": [1,2,3]}'
        ]
        
        objects = []
        async for obj in helper.process_stream(self.generate_chunks(chunks)):
            objects.append(obj)
            
        assert len(objects) == 1
        assert objects[0] == {
            "outer": {"inner": {"value": 42}},
            "array": [1,2,3]
        }

    @pytest.mark.asyncio
    async def test_validation(self, helper):
        """Test JSON validation"""
        def validator(obj):
            return isinstance(obj.get('value'), int)
            
        chunks = [
            '{"value": 42}',
            '{"value": "string"}',
            '{"value": 100}'
        ]
        
        objects = []
        async for obj in helper.process_stream(
            self.generate_chunks(chunks),
            validator=validator
        ):
            objects.append(obj)
            
        assert len(objects) == 2
        assert all(obj['value'] in [42, 100] for obj in objects)

    @pytest.mark.asyncio
    async def test_repair_options(self, helper):
        """Test JSON repair options"""
        chunks = [
            '{key: "value",',  # Missing quotes around key
            ' trailing: true,}',  # Trailing comma
        ]
        
        # Test with repair enabled
        objects_with_repair = []
        async for obj in helper.process_stream(
            self.generate_chunks(chunks),
            repair_json=True
        ):
            objects_with_repair.append(obj)
            
        assert len(objects_with_repair) == 1
        assert objects_with_repair[0] == {"key": "value", "trailing": True}
        
        # Test with repair disabled
        objects_without_repair = []
        async for obj in helper.process_stream(
            self.generate_chunks(chunks),
            repair_json=False
        ):
            objects_without_repair.append(obj)
            
        assert len(objects_without_repair) == 0

    @pytest.mark.asyncio
    async def test_template_validation(self):
        """Test JSON template validation"""
        template = {
            "name": "string",
            "value": "number"
        }
        
        helper = JSONStreamHelper(template=template)
        
        chunks = [
            '{"name": "test", "value": 42}',
            '{"name": "valid", "value": 100}'
        ]
        
        objects = []
        async for obj in helper.process_stream(self.generate_chunks(chunks)):
            objects.append(obj)
            
        assert len(objects) == 2
        assert all(
            isinstance(obj['name'], str) and 
            isinstance(obj['value'], (int, float))
            for obj in objects
        )

    @pytest.mark.asyncio
    async def test_error_handling(self, helper):
        """Test error handling for invalid JSON"""
        chunks = [
            '{"invalid": json',
            'content} {"valid": true}'
        ]
        
        objects = []
        async for obj in helper.process_stream(
            self.generate_chunks(chunks),
            repair_json=False  # Don't repair to test error handling
        ):
            objects.append(obj)
            
        # Should only get the valid object
        assert len(objects) == 1
        assert objects[0] == {"valid": True}

    @pytest.mark.asyncio
    async def test_buffer_overflow(self, helper):
        """Test handling of buffer overflow"""
        # Create a very large invalid JSON string
        large_chunk = '{' * 10001  # Exceeds default max_buffer_size
        
        objects = []
        async for obj in helper.process_stream(
            self.generate_chunks([large_chunk])
        ):
            objects.append(obj)
        
        # Should not process anything due to buffer overflow
        assert len(objects) == 0
  