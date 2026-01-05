import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import tempfile
from pathlib import Path

from open_data_pvnet.scripts.upload_eia_to_s3 import upload_file, check_bucket_access, upload_directory_to_s3

class TestS3Upload(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("test content")

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_upload_file_dry_run(self):
        """Test upload_file with dry_run=True"""
        mock_client = MagicMock()
        result = upload_file(
            mock_client, 
            self.test_file, 
            "my-bucket", 
            "key", 
            dry_run=True
        )
        self.assertTrue(result)
        mock_client.upload_file.assert_not_called()

    def test_upload_file_real(self):
        """Test upload_file with dry_run=False"""
        mock_client = MagicMock()
        result = upload_file(
            mock_client, 
            self.test_file, 
            "my-bucket", 
            "key", 
            dry_run=False
        )
        self.assertTrue(result)
        mock_client.upload_file.assert_called_once()    

    def test_check_bucket_access_success(self):
        mock_client = MagicMock()
        result = check_bucket_access(mock_client, "my-bucket")
        self.assertTrue(result)
        mock_client.head_bucket.assert_called_with(Bucket="my-bucket")

    def test_check_bucket_access_dry_run(self):
        result = check_bucket_access(None, "my-bucket") # None client implies dry_run in usage
        self.assertTrue(result)

    @patch("open_data_pvnet.scripts.upload_eia_to_s3.get_s3_client")
    def test_upload_directory(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        result = upload_directory_to_s3(
            self.test_dir,
            "my-bucket",
            "prefix",
            dry_run=False
        )
        
        self.assertTrue(result)
        # Should be called for the one file in temp dir
        mock_client.upload_file.assert_called()

if __name__ == '__main__':
    unittest.main()
