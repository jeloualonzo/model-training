# filepath: python/s3_uploader.py
"""
AWS S3 Uploader for TF.js Models
Uploads converted models to S3 bucket
"""
import os
import boto3
from pathlib import Path
from typing import Dict, List
from botocore.exceptions import ClientError
class S3Uploader:
    def __init__(
        self,
        aws_access_key: str,
        aws_secret_key: str,
        bucket_name: str,
        region: str = 'ap-southeast-1'
    ):
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        print(f"✅ S3 Uploader initialized: s3://{bucket_name}")
    
    def upload_tfjs_model(
        self,
        tfjs_dir: str,
        model_uuid: str,
        files_dict: Dict[str, any]
    ) -> Dict[str, str]:
        """
        Upload TF.js model files to S3
        
        ✅ SIMPLIFIED: models/classifier/{uuid}/model.json
        
        Args:
            tfjs_dir: Local directory containing TF.js model
            model_uuid: Unique identifier for this model
            files_dict: Dictionary from tfjs_converter.get_model_files()
        
        Returns:
            Dictionary with S3 keys and URLs
        """
        try:
            # ✅ SIMPLIFIED PATH: models/classifier/{uuid}/
            s3_base_key = f"models/classifier/{model_uuid}"
            uploaded_files = {}
            
            print(f"\n📤 Uploading TF.js model to S3...")
            print(f"   Bucket: {self.bucket_name}")
            print(f"   Path: {s3_base_key}/")
            
            # Upload model.json
            model_json_key = f"{s3_base_key}/model.json"
            self._upload_file(
                files_dict['model_json'],
                model_json_key,
                'application/json'
            )
            uploaded_files['model_json'] = model_json_key
            
            # Upload metadata.json
            metadata_key = f"{s3_base_key}/metadata.json"
            self._upload_file(
                files_dict['metadata'],
                metadata_key,
                'application/json'
            )
            uploaded_files['metadata'] = metadata_key
            
            # Upload weight shards
            uploaded_files['weights'] = []
            for idx, weight_file in enumerate(files_dict['weights']):
                weight_filename = Path(weight_file).name
                weight_key = f"{s3_base_key}/{weight_filename}"
                self._upload_file(
                    weight_file,
                    weight_key,
                    'application/octet-stream'
                )
                uploaded_files['weights'].append(weight_key)
                print(f"   ✅ Uploaded weight shard {idx + 1}/{len(files_dict['weights'])}")
            
            # Generate public URLs
            base_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com"
            uploaded_files['base_url'] = base_url
            uploaded_files['model_url'] = f"{base_url}/{model_json_key}"
            uploaded_files['metadata_url'] = f"{base_url}/{metadata_key}"
            
            print(f"\n✅ Upload complete!")
            print(f"   📍 S3 Key: {model_json_key}")
            print(f"   🌐 Public URL: {uploaded_files['model_url']}")
            
            return uploaded_files
            
        except Exception as e:
            print(f"❌ S3 upload failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _upload_file(self, file_path: str, s3_key: str, content_type: str):
        """
        ✅ FIXED: Upload without ACL (bucket policy handles public access)
        """
        try:
            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'CacheControl': 'max-age=31536000'  # 1 year cache
                    # ✅ REMOVED: 'ACL': 'public-read'
                }
            )
            print(f"   ✅ Uploaded: {s3_key}")
        except ClientError as e:
            print(f"   ❌ Failed to upload {s3_key}: {e}")
            raise
    
    def delete_model(self, model_uuid: str):
        """Delete TF.js model from S3"""
        try:
            s3_base_key = f"models/classifier/{model_uuid}"
            
            # List all objects with this prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=s3_base_key
            )
            
            if 'Contents' not in response:
                print(f"⚠️ No files found for model {model_uuid}")
                return
            
            # Delete all objects
            objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
            
            self.s3_client.delete_objects(
                Bucket=self.bucket_name,
                Delete={'Objects': objects_to_delete}
            )
            
            print(f"✅ Deleted {len(objects_to_delete)} files for model {model_uuid}")
            
        except Exception as e:
            print(f"❌ Failed to delete model from S3: {e}")
            raise
