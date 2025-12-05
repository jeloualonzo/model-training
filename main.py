# filepath: python/main.py 
"""
Flask API Server with S3 Operations
"""
import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed 
import io
from PIL import Image
import uuid
from supabase_recorder import SupabaseRecorder
from io import BytesIO
import base64
import cv2 
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import multiprocessing
import os
import json
from datetime import datetime, timedelta, timezone 
import threading
import time
from threading import Lock
from config import config
from training import SignatureTrainer
from typing import List, Dict
import hashlib
from typing import Optional
from utils import (
    get_latest_model, 
    get_model_metadata, 
    list_trained_models,
    validate_training_data,
    format_training_summary
)
# ===========================
# IN-MEMORY COUNTDOWN SYSTEM
# ===========================
class CountdownTimer:
    """In-memory countdown timer for a single admin"""
    
    def __init__(self, admin_id: str, duration_seconds: int, 
                 student_count: int, signature_count: int, trigger_reason: str = 'manual'):
        self.admin_id = admin_id
        self.duration_seconds = duration_seconds
        self.remaining_seconds = duration_seconds
        self.student_count = student_count
        self.signature_count = signature_count
        self.trigger_reason = trigger_reason
        self.status = 'counting'  # counting, paused, completed, cancelled
        self.created_at = datetime.now(timezone.utc)
        self.started_at = datetime.now(timezone.utc)
        self.paused_at = None
        self.lock = Lock()
        self.timer_thread = None
        self._stop_event = threading.Event()
        self.schedule_id = None
        
    def start(self, on_complete_callback):
        """Start the countdown timer"""
        self.on_complete = on_complete_callback
        self.timer_thread = threading.Thread(target=self._run_countdown, daemon=True)
        self.timer_thread.start()
        print(f"⏱️  Started countdown for admin {self.admin_id}: {self.duration_seconds}s")
    
    def _run_countdown(self):
        """Internal countdown loop"""
        while self.remaining_seconds > 0 and not self._stop_event.is_set():
            time.sleep(1)
            
            with self.lock:
                if self.status == 'counting':
                    self.remaining_seconds -= 1
                    
                    # Log every 10 seconds
                    if self.remaining_seconds % 10 == 0:
                        print(f"⏱️  Admin {self.admin_id}: {self.remaining_seconds}s remaining")
        
        # Countdown finished
        if not self._stop_event.is_set() and self.remaining_seconds == 0:
            with self.lock:
                self.status = 'completed'
            print(f"⏰ Countdown completed for admin {self.admin_id}")
            self.on_complete(self.admin_id)
    
    def pause(self) -> bool:
        """Pause the countdown"""
        with self.lock:
            if self.status == 'counting':
                self.status = 'paused'
                self.paused_at = datetime.now(timezone.utc)
                print(f"⏸️  Paused countdown for admin {self.admin_id} at {self.remaining_seconds}s")
                return True
        return False
    
    def resume(self) -> bool:
        """Resume the countdown"""
        with self.lock:
            if self.status == 'paused':
                self.status = 'counting'
                self.paused_at = None
                print(f"▶️  Resumed countdown for admin {self.admin_id} with {self.remaining_seconds}s")
                return True
        return False
    
    def adjust(self, adjustment_seconds: int) -> bool:
        """Adjust remaining time (can be positive or negative)"""
        with self.lock:
            old_remaining = self.remaining_seconds
            self.remaining_seconds = max(1, self.remaining_seconds + adjustment_seconds)
            print(f"🔧 Adjusted countdown for admin {self.admin_id}: {old_remaining}s → {self.remaining_seconds}s")
            return True
    
    def cancel(self):
        """Cancel the countdown"""
        with self.lock:
            self.status = 'cancelled'
        self._stop_event.set()
        print(f"🛑 Cancelled countdown for admin {self.admin_id}")
    
    def get_state(self) -> dict:
        """Get current countdown state"""
        with self.lock:
            return {
                'admin_id': self.admin_id,
                'status': self.status,
                'remaining_seconds': self.remaining_seconds,
                'duration_seconds': self.duration_seconds,
                'student_count': self.student_count,
                'signature_count': self.signature_count,
                'trigger_reason': self.trigger_reason,
                'created_at': self.created_at.isoformat(),
                'started_at': self.started_at.isoformat(),
                'paused_at': self.paused_at.isoformat() if self.paused_at else None
            }


class CountdownManager:
    """Manages all active countdown timers"""
    
    def __init__(self):
        self.timers: Dict[str, CountdownTimer] = {}
        self.lock = Lock()
    
    def start_countdown(self, admin_id: str, duration_seconds: int,
                       student_count: int, signature_count: int, 
                       trigger_reason: str = 'manual') -> CountdownTimer:
        """Start a new countdown for an admin"""
        with self.lock:
            # ✅ Cancel existing countdown if any (RESET)
            if admin_id in self.timers:
                old_timer = self.timers[admin_id]
                old_timer.cancel()
                del self.timers[admin_id]
                print(f"🔄 RESET: Replaced existing countdown for admin {admin_id}")
                print(f"   Old remaining: {old_timer.remaining_seconds}s")
                print(f"   New duration: {duration_seconds}s")
            
            # Create new timer
            timer = CountdownTimer(
                admin_id=admin_id,
                duration_seconds=duration_seconds,
                student_count=student_count,
                signature_count=signature_count,
                trigger_reason=trigger_reason
            )
            
            # ✅ Write to database IMMEDIATELY
            self._write_countdown_start_to_db(admin_id, timer)
            
            # Start timer with completion callback
            timer.start(on_complete_callback=self._on_countdown_complete)
            
            self.timers[admin_id] = timer
            return timer

    def _write_countdown_start_to_db(self, admin_id: str, timer: CountdownTimer):
        """Write countdown START to database (so frontend can see it)"""
        try:
            from supabase import create_client
            supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
            
            now = datetime.now(timezone.utc)
            countdown_ends_at = now + timedelta(seconds=timer.duration_seconds)
            
            # Generate unique schedule_id
            import uuid
            schedule_id = str(uuid.uuid4())
            
            # ✅ CRITICAL FIX: Use timer.trigger_reason (from the timer object)
            record = {
                'schedule_id': schedule_id,
                'admin_id': admin_id,
                'countdown_minutes': int(timer.duration_seconds / 60),
                'countdown_started_at': timer.started_at.isoformat(),
                'countdown_ends_at': countdown_ends_at.isoformat(),
                'status': 'counting',
                'snapshot_student_count': timer.student_count,
                'snapshot_signature_count': timer.signature_count,
                'trigger_reason': timer.trigger_reason,  # ✅ This should have the correct reason
                'updated_at': now.isoformat()
            }
            
            print(f"📋 Inserting countdown to DB:")
            print(f"   Admin: {admin_id}")
            print(f"   Trigger reason: {timer.trigger_reason}")  # ✅ Debug log
            
            # Delete old countdowns for this admin
            supabase.table('training_schedule')\
                .delete()\
                .eq('admin_id', admin_id)\
                .in_('status', ['counting', 'paused', 'idle'])\
                .execute()
            
            # Insert new countdown
            result = supabase.table('training_schedule').insert(record).execute()
            
            if result.data:
                print(f"✅ Wrote countdown START to DB (status: counting, reason: {timer.trigger_reason})")
                timer.schedule_id = result.data[0]['schedule_id'] if result.data else schedule_id
            else:
                print(f"⚠️ Insert succeeded but no data returned")
                timer.schedule_id = schedule_id
            
        except Exception as e:
            print(f"⚠️ Failed to write countdown START to DB: {e}")
            import traceback
            traceback.print_exc()

    def get_countdown(self, admin_id: str) -> Optional[CountdownTimer]:
        """Get countdown for an admin"""
        with self.lock:
            return self.timers.get(admin_id)
    
    def pause_countdown(self, admin_id: str) -> bool:
        """Pause countdown for an admin"""
        timer = self.get_countdown(admin_id)
        if timer:
            success = timer.pause()
            if success:
                # ✅ Write pause to database
                self._update_countdown_status_in_db(admin_id, 'paused', timer)
            return success
        return False
    def resume_countdown(self, admin_id: str) -> bool:
        """Resume countdown for an admin"""
        timer = self.get_countdown(admin_id)
        if timer:
            success = timer.resume()
            if success:
                # ✅ Write resume to database
                self._update_countdown_status_in_db(admin_id, 'counting', timer)
            return success
        return False
    def adjust_countdown(self, admin_id: str, adjustment_seconds: int) -> bool:
        """Adjust countdown time"""
        timer = self.get_countdown(admin_id)
        if timer:
            success = timer.adjust(adjustment_seconds)
            if success:
                # ✅ Write adjustment to database
                self._update_countdown_status_in_db(admin_id, timer.status, timer)
            return success
        return False
    # ✅ ADD THIS NEW METHOD:
    def _update_countdown_status_in_db(self, admin_id: str, status: str, timer: CountdownTimer):
        """Update countdown status in database"""
        try:
            from supabase import create_client
            supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
            
            now = datetime.now(timezone.utc)
            
            # Calculate new countdown_ends_at if status is counting
            if status == 'counting':
                countdown_ends_at = now + timedelta(seconds=timer.remaining_seconds)
                update_data = {
                    'status': status,
                    'countdown_ends_at': countdown_ends_at.isoformat(),
                    'updated_at': now.isoformat()
                }
            else:
                update_data = {
                    'status': status,
                    'updated_at': now.isoformat()
                }
            
            supabase.table('training_schedule')\
                .update(update_data)\
                .eq('admin_id', admin_id)\
                .eq('status', 'counting' if status == 'paused' else 'paused')\
                .execute()
            
            print(f"✅ Updated countdown status in DB: {status}")
            
        except Exception as e:
            print(f"⚠️ Failed to update countdown status: {e}")

    def cancel_countdown(self, admin_id: str) -> bool:
        """Cancel countdown for an admin"""
        with self.lock:
            if admin_id in self.timers:
                timer = self.timers[admin_id]
                timer.cancel()
                del self.timers[admin_id]
                
                # Write cancellation to DB
                self._write_countdown_to_db(admin_id, 'cancelled', timer)
                return True
            else:
                # ✅ ADD THIS: Handle case where timer is not in memory but exists in DB
                print(f"⚠️ No in-memory timer for {admin_id}, checking DB...")
                try:
                    from supabase import create_client
                    supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
                    
                    # Try to cancel any active countdown in DB
                    result = supabase.table('training_schedule')\
                        .update({
                            'status': 'cancelled',
                            'updated_at': datetime.now(timezone.utc).isoformat()
                        })\
                        .eq('admin_id', admin_id)\
                        .in_('status', ['counting', 'paused'])\
                        .execute()
                    
                    if result.data and len(result.data) > 0:
                        print(f"✅ Cancelled countdown in DB for {admin_id}")
                        return True
                except Exception as e:
                    print(f"❌ Error cancelling in DB: {e}")
                
                return False
    
    def complete_countdown(self, admin_id: str) -> bool:
        """Manually complete countdown (skip to end)"""
        with self.lock:
            if admin_id in self.timers:
                timer = self.timers[admin_id]
                timer.cancel()  # Stop the timer thread
                del self.timers[admin_id]
                
                # Trigger training immediately
                self._on_countdown_complete(admin_id, write_db=True)
                return True
        return False
    
    def _on_countdown_complete(self, admin_id: str, write_db: bool = True):
        """Callback when countdown finishes"""
        print(f"\n⏰ Countdown completed for admin {admin_id}")
        
        # Get timer before removing it
        timer = None
        with self.lock:
            timer = self.timers.get(admin_id)
            if admin_id in self.timers:
                del self.timers[admin_id]
        
        # Update DB to 'completed'
        if write_db and timer:
            self._update_countdown_to_completed(admin_id, timer)
        
        print(f"✅ Countdown completed - frontend will handle training")


    # ✅ ADD THIS NEW METHOD:
    def _update_countdown_to_completed(self, admin_id: str, timer: CountdownTimer):
        """Update existing countdown record to 'completed' status"""
        try:
            from supabase import create_client
            supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
            
            now = datetime.now(timezone.utc)
            
            # ✅ UPDATE existing record (don't insert new)
            result = supabase.table('training_schedule')\
                .update({
                    'status': 'completed',
                    'countdown_ends_at': now.isoformat(),
                    'updated_at': now.isoformat()
                })\
                .eq('admin_id', admin_id)\
                .in_('status', ['counting', 'paused'])\
                .execute()
            
            if result.data:
                print(f"✅ Updated countdown record to 'completed' (admin: {admin_id})")
            else:
                print(f"⚠️ No active countdown found to update for admin {admin_id}")
            
        except Exception as e:
            print(f"⚠️ Failed to update countdown to completed: {e}")
            import traceback
            traceback.print_exc()


    # ✅ ALSO UPDATE: _write_countdown_to_db (used by cancel)
    def _write_countdown_to_db(self, admin_id: str, status: str, timer: CountdownTimer):
        """Write countdown record to database (for history/audit)"""
        try:
            from supabase import create_client
            supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
            
            now = datetime.now(timezone.utc)
            
            if status == 'cancelled':
                # ✅ UPDATE existing record to cancelled
                result = supabase.table('training_schedule')\
                    .update({
                        'status': 'cancelled',
                        'countdown_ends_at': None,
                        'updated_at': now.isoformat()
                    })\
                    .eq('admin_id', admin_id)\
                    .in_('status', ['counting', 'paused'])\
                    .execute()
                
                if result.data:
                    print(f"✅ Updated countdown to 'cancelled' (admin: {admin_id})")
                else:
                    print(f"⚠️ No active countdown found to cancel for admin {admin_id}")
            
        except Exception as e:
            print(f"⚠️ Failed to write countdown to DB: {e}")

# Global countdown manager
countdown_manager = CountdownManager()
download_thread_pool = ThreadPoolExecutor(max_workers=50) 
recent_training_requests = {}  # {admin_id: timestamp}
training_request_lock = threading.Lock()

def start_training_background(admin_id, student_ids, job_id):
    """
    ✅ UPDATED: Save ACTUAL training data counts (not snapshot)
    """
    try:
        from supabase import create_client
        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        print(f"\n🚀 Training: {len(student_ids)} students (Admin: {admin_id}, Job: {job_id})")
        
        # ✅ STEP 1: Load default signatures
        print("\n📥 Loading default signatures from S3...")
        default_signatures = load_default_signatures_from_s3()
        
        if not default_signatures:
            print("❌ Failed to load default signatures")
            supabase.table('training_jobs').update({
                'status': 'failed',
                'error_message': 'Failed to load default signatures',
                'completed_at': datetime.now(timezone.utc).isoformat()
            }).eq('job_id', job_id).execute()
            return
        
        unknown_images = default_signatures.get('unknown', [])
        non_sig_images = default_signatures.get('non-signature', [])
        
        print(f"✅ Loaded {len(unknown_images)} unknown, {len(non_sig_images)} non-signature defaults")
        
        # Initialize S3 client
        s3 = get_s3_client()
        if not s3:
            print("❌ S3 not available")
            supabase.table('training_jobs').update({
                'status': 'failed',
                'error_message': 'S3 not available',
                'completed_at': datetime.now(timezone.utc).isoformat()
            }).eq('job_id', job_id).execute()
            return
        
        bucket = config.S3['bucket']
        
        # ✅ STEP 2: Fetch student data
        student_classes = []
        all_s3_keys = []
        student_key_mapping = {}
        
        for school_student_id in student_ids:
            try:
                student_response = supabase.table('students')\
                    .select('*')\
                    .eq('student_id', school_student_id)\
                    .eq('admin_id', admin_id)\
                    .limit(1)\
                    .execute()
                
                if not student_response.data or len(student_response.data) == 0:
                    print(f"   ⚠️ No student found for {school_student_id}")
                    continue
                
                student_data = student_response.data[0]
                
                signatures_response = supabase.table('student_signatures')\
                    .select('s3_key, type')\
                    .eq('school_student_id', school_student_id)\
                    .execute()
                
                if not signatures_response.data or len(signatures_response.data) < 3:
                    print(f"   ⚠️ Student {school_student_id} has only {len(signatures_response.data) if signatures_response.data else 0} signatures")
                    continue
                
                uploaded_keys = [sig['s3_key'] for sig in signatures_response.data if sig['type'] == 'uploaded']
                captured_keys = [sig['s3_key'] for sig in signatures_response.data if sig['type'] == 'captured']
                
                all_s3_keys.extend(uploaded_keys)
                all_s3_keys.extend(captured_keys)
                
                student_key_mapping[school_student_id] = {
                    'student_data': student_data,
                    'uploaded_keys': uploaded_keys,
                    'captured_keys': captured_keys
                }
                
                print(f"   ✅ {school_student_id}: {len(uploaded_keys)} uploaded, {len(captured_keys)} captured")
                
            except Exception as e:
                print(f"   ❌ Error processing {school_student_id}: {e}")
                continue
        
        if len(student_key_mapping) < 2:
            print(f"❌ Only {len(student_key_mapping)} students loaded (need 2+)")
            supabase.table('training_jobs').update({
                'status': 'failed',
                'error_message': f'Only {len(student_key_mapping)} students available',
                'completed_at': datetime.now(timezone.utc).isoformat()
            }).eq('job_id', job_id).execute()
            return
        
        # ✅ STEP 3: Save ACTUAL training data (THIS IS THE FIX)
        actual_student_count = len(student_key_mapping)
        actual_signature_count = len(all_s3_keys)
        
        print(f"\n💾 Saving ACTUAL training data to job:")
        print(f"   Students being trained: {actual_student_count}")
        print(f"   Signatures being trained: {actual_signature_count}")
        
        supabase.table('training_jobs').update({
            'training_start_student_count': actual_student_count,
            'training_start_signature_count': actual_signature_count
        }).eq('job_id', job_id).execute()
        
        # ✅ STEP 4: Download signatures in parallel
        print(f"\n📥 Downloading {len(all_s3_keys)} signatures...")
        download_results = download_signature_batch(s3, bucket, all_s3_keys)
        image_map = {s3_key: img_data for s3_key, img_data in download_results if img_data is not None}
        
        # ✅ STEP 5: Organize by student
        for school_student_id, data in student_key_mapping.items():
            student_data = data['student_data']
            uploaded_keys = data['uploaded_keys']
            captured_keys = data['captured_keys']
            
            all_samples = []
            
            for s3_key in uploaded_keys:
                if s3_key in image_map:
                    all_samples.append({
                        'thumbnail': image_map[s3_key],
                        'timestamp': int(time.time()),
                        'type': 'uploaded'
                    })
            
            for s3_key in captured_keys:
                if s3_key in image_map:
                    all_samples.append({
                        'thumbnail': image_map[s3_key],
                        'timestamp': int(time.time()),
                        'type': 'captured'
                    })
            
            if len(all_samples) >= 3:
                student_classes.append({
                    'student': {
                        'id': student_data['id'],
                        'student_id': student_data['student_id'],
                        'firstname': student_data['firstname'],
                        'middlename': student_data.get('middlename'),
                        'surname': student_data['surname'],
                        'program': student_data['program'],
                        'year': student_data['year']
                    },
                    'samples': all_samples,
                    'isDefault': False
                })
        
        # ✅ STEP 6: Add default classes
        if len(unknown_images) > 0:
            student_classes.append({
                'student': {
                    'id': 0,
                    'student_id': 'UNKNOWN',
                    'firstname': 'Unknown',
                    'surname': 'Signature',
                    'program': 'Default',
                    'year': 'N/A'
                },
                'samples': [
                    {'thumbnail': img, 'timestamp': int(time.time()), 'type': 'uploaded'}
                    for img in unknown_images
                ],
                'isDefault': True,
                'defaultName': 'unknown'
            })
            print(f"✅ Added UNKNOWN class with {len(unknown_images)} samples")
        
        if len(non_sig_images) > 0:
            student_classes.append({
                'student': {
                    'id': 0,
                    'student_id': 'NON-SIGNATURE',
                    'firstname': 'Non',
                    'surname': 'Signature',
                    'program': 'Default',
                    'year': 'N/A'
                },
                'samples': [
                    {'thumbnail': img, 'timestamp': int(time.time()), 'type': 'uploaded'}
                    for img in non_sig_images
                ],
                'isDefault': True,
                'defaultName': 'non-signature'
            })
            print(f"✅ Added NON-SIGNATURE class with {len(non_sig_images)} samples")
        
        # Validate training data
        print(f"\n📊 Training data summary:")
        print(f"   Total classes: {len(student_classes)}")
        print(f"   Student classes: {len([c for c in student_classes if not c.get('isDefault')])}")
        print(f"   Default classes: {len([c for c in student_classes if c.get('isDefault')])}")
        
        if len(student_classes) < 3:
            print(f"❌ Not enough classes: {len(student_classes)}")
            supabase.table('training_jobs').update({
                'status': 'failed',
                'error_message': f'Not enough classes: {len(student_classes)}',
                'completed_at': datetime.now(timezone.utc).isoformat()
            }).eq('job_id', job_id).execute()
            return
        
        # ✅ STEP 7: Start actual training
        print(f"\n🎯 Starting training with {len(student_classes)} classes...")
        
        trainer = SignatureTrainer()
        metrics = trainer.train(classes=student_classes)
        
        # ✅ STEP 8: Handle deployment
        print("\n💾 Saving and deploying model...")
        save_result = trainer.save_model_with_tfjs()
        
        # ✅ STEP 9: Record to Supabase
        if config.SUPABASE['enabled']:
            from supabase_recorder import SupabaseRecorder
            recorder = SupabaseRecorder(
                config.SUPABASE['url'],
                config.SUPABASE['key']
            )
            
            record_id = recorder.record_training_start(
                model_uuid=save_result['model_uuid'],
                sample_count=metrics['total_samples'],
                student_count=len([c for c in student_classes if not c.get('isDefault')]),
                genuine_count=metrics['total_samples'],
                admin_id=admin_id
            )
            
            print(f"✅ Training record created: ID {record_id}")
            
            recorder.update_training_complete(
                record_id=record_id,
                metrics=metrics,
                s3_info=save_result.get('s3_info', {}),
                keras_model_path=save_result['keras_model_path'],
                tfjs_model_path=save_result.get('tfjs_model_path', ''),
                class_info=metrics.get('class_info')
            )
            
            print(f"✅ Training record completed: ID {record_id}")
        
        # ✅ STEP 10: Update job as completed
        supabase.table('training_jobs').update({
            'status': 'completed',
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'model_uuid': save_result['model_uuid']
        }).eq('job_id', job_id).execute()
        
        print(f"\n✅ Training completed successfully!")
        print(f"   Accuracy: {metrics['validation_accuracy']:.2%}")
        print(f"   Model UUID: {save_result['model_uuid']}")
        
        # ✅ STEP 11: Check if retrain was queued during training
        print("\n🔍 Checking for queued retrain...")
        
        job_check = supabase.table('training_jobs')\
            .select('pending_retrain_check, training_start_student_count, training_start_signature_count')\
            .eq('job_id', job_id)\
            .single()\
            .execute()
        
        if job_check.data and job_check.data.get('pending_retrain_check'):
            print('📝 Retrain was queued during training - checking if needed...')
            
            # Wait 10 seconds (shorter grace period)
            print('⏳ Waiting 10 seconds (grace period)...')
            time.sleep(10)
            
            # Get current training data
            try:
                current_students_response = supabase.table('students')\
                    .select('student_id', count='exact')\
                    .eq('admin_id', admin_id)\
                    .execute()
                
                current_student_count = current_students_response.count or 0
                
                # Count signatures for these students
                student_ids_for_count = [s['student_id'] for s in current_students_response.data] if current_students_response.data else []
                
                if student_ids_for_count:
                    current_signatures_response = supabase.table('student_signatures')\
                        .select('id', count='exact')\
                        .in_('school_student_id', student_ids_for_count)\
                        .execute()
                    
                    current_signature_count = current_signatures_response.count or 0
                else:
                    current_signature_count = 0
                
                # Get training start data
                training_start_students = job_check.data.get('training_start_student_count', 0)
                training_start_signatures = job_check.data.get('training_start_signature_count', 0)
                
                print(f'📊 Data comparison:')
                print(f'   Training start: {training_start_students} students, {training_start_signatures} signatures')
                print(f'   Current data: {current_student_count} students, {current_signature_count} signatures')
                
                # Check if data changed
                students_changed = current_student_count > training_start_students
                signatures_changed = current_signature_count > training_start_signatures
                
                if students_changed or signatures_changed:
                    # Build trigger reason
                    trigger_reasons = []
                    
                    if students_changed:
                        diff = current_student_count - training_start_students
                        trigger_reasons.append(f'{diff} new student{"s" if diff > 1 else ""} added during training')
                    
                    if signatures_changed:
                        diff = current_signature_count - training_start_signatures
                        trigger_reasons.append(f'{diff} new signature{"s" if diff > 1 else ""} added during training')
                    
                    trigger_reason = '; '.join(trigger_reasons)
                    
                    print(f'✅ Data changed - starting countdown')
                    print(f'   Reason: {trigger_reason}')
                    
                    # Start countdown (2 minutes since changes are fresh)
                    start_countdown_for_admin(
                        admin_id=admin_id,
                        countdown_minutes=2,
                        trigger_reason=trigger_reason
                    )
                else:
                    print('✅ No data changes - false alarm')
                
            except Exception as e:
                print(f'❌ Error comparing data: {e}')
                import traceback
                traceback.print_exc()
            
            # Clear the queue flag
            supabase.table('training_jobs').update({
                'pending_retrain_check': False
            }).eq('job_id', job_id).execute()
        else:
            print('✅ No retrain queued')
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Update job as failed
        try:
            supabase.table('training_jobs').update({
                'status': 'failed',
                'error_message': str(e),
                'completed_at': datetime.now(timezone.utc).isoformat()
            }).eq('job_id', job_id).execute()
        except:
            pass
# ===========================
# PARALLEL DOWNLOAD HELPER
# ===========================
def download_signature_batch(s3_client, bucket, s3_keys, max_workers=50):
    """
    Download and process S3 objects in parallel (optimized for T4 GPU)
    
    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        s3_keys: List of S3 keys to download
        max_workers: Download workers (default 50 for T4 GPU)
    
    Returns:
        List of tuples: (s3_key, img_base64) or (s3_key, None) on error
    """
    
    # STEP 1: Parallel download (I/O-bound)
    def download_raw(s3_key):
        try:
            response = s3_client.get_object(Bucket=bucket, Key=s3_key)
            return (s3_key, response['Body'].read())
        except Exception as e:
            print(f"   ❌ Download failed {s3_key}: {e}")
            return (s3_key, None)
    
    print(f"   🔧 Downloading {len(s3_keys)} files with {max_workers} workers...")
    download_start = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        raw_data = list(executor.map(download_raw, s3_keys))
    
    download_time = time.time() - download_start
    print(f"   ⏱️  Download: {download_time:.2f}s ({len(s3_keys)/download_time:.1f} files/sec)")
    
    # STEP 2: Parallel processing (CPU-bound)
    def process_image(item):
        s3_key, img_bytes = item
        if img_bytes is None:
            return (s3_key, None)
        
        try:
            img = Image.open(io.BytesIO(img_bytes))
            
            if img.mode != 'L':
                img = img.convert('L')
            
            img = img.resize((224, 224), Image.LANCZOS)
            
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return (s3_key, f"data:image/png;base64,{img_base64}")
        except Exception as e:
            print(f"   ❌ Process failed {s3_key}: {e}")
            return (s3_key, None)
    
    cpu_workers = multiprocessing.cpu_count()
    print(f"   🔧 Processing {len(raw_data)} images with {cpu_workers} CPU workers...")
    process_start = time.time()
    
    with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
        results = list(executor.map(process_image, raw_data))
    
    process_time = time.time() - process_start
    print(f"   ⏱️  Processing: {process_time:.2f}s ({len(raw_data)/process_time:.1f} images/sec)")
    print(f"   📊 Total: {download_time + process_time:.2f}s")
    
    return results


def start_countdown_for_admin(admin_id: str, countdown_minutes: int, trigger_reason: str):
    """
    Start a countdown for the given admin.
    Used by backend to automatically trigger countdowns.
    """
    try:
        print(f'🚀 Starting countdown for admin: {admin_id}')  # ✅ Use print
        print(f'   Duration: {countdown_minutes} minutes')
        print(f'   Reason: {trigger_reason}')
        
        from supabase import create_client
        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        # Get current training data snapshot
        eligible_students_response = supabase.table('students')\
            .select('student_id', count='exact')\
            .eq('admin_id', admin_id)\
            .execute()
        
        if not eligible_students_response.data or len(eligible_students_response.data) < 2:
            print(f'⚠️ Not enough students for admin {admin_id} - skipping countdown')
            return False
        
        student_count = len(eligible_students_response.data)
        
        # Count signatures
        student_ids = [s['student_id'] for s in eligible_students_response.data]
        
        signatures_response = supabase.table('student_signatures')\
            .select('id', count='exact')\
            .in_('school_student_id', student_ids)\
            .execute()
        
        signature_count = signatures_response.count or 0
        
        # Delete any existing active countdowns
        supabase.table('training_schedule')\
            .delete()\
            .eq('admin_id', admin_id)\
            .in_('status', ['counting', 'paused', 'idle'])\
            .execute()
        
        # Create new countdown
        import uuid
        schedule_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        countdown_ends_at = now + timedelta(minutes=countdown_minutes)
        
        record = {
            'schedule_id': schedule_id,
            'admin_id': admin_id,
            'countdown_minutes': countdown_minutes,
            'countdown_started_at': now.isoformat(),
            'countdown_ends_at': countdown_ends_at.isoformat(),
            'status': 'counting',
            'snapshot_student_count': student_count,
            'snapshot_signature_count': signature_count,
            'trigger_reason': trigger_reason,
            'updated_at': now.isoformat()
        }
        
        result = supabase.table('training_schedule').insert(record).execute()
        
        if result.data:
            print(f'✅ Countdown started successfully: {schedule_id}')
            return True
        else:
            print('❌ Failed to create countdown record')
            return False
            
    except Exception as e:
        print(f'❌ Error starting countdown: {e}')
        import traceback
        traceback.print_exc()
        return False
# Initialize Flask app
app = Flask(__name__)

# ✅ Increase max upload size to 500MB
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB


# Configure CORS - Allow all origins
CORS(app, resources={
    r"/*": {
        "origins": config.API['cors_origins'],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # ✅ ADD PUT
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
# ===========================
# GLOBAL STATE
# ===========================
training_jobs = {}  # {admin_id: {is_training, progress, ...}}
training_jobs_lock = threading.Lock()  # Thread-safe access to training_jobs
supabase_recorder = None
# ===========================
# S3 CLIENT INITIALIZATION
# ===========================
def get_s3_client():
    """Initialize S3 client with error handling and optimized connection pool"""
    if not config.S3['enabled']:
        return None
    
    try:
        from botocore.config import Config
        
        # ✅ Configure connection pool to match thread pool size
        boto_config = Config(
            max_pool_connections=100,  # Match or exceed max_workers (50 + buffer)
            retries={
                'max_attempts': 3,
                'mode': 'standard'
            }
        )
        
        return boto3.client(
            's3',
            region_name=config.S3['region'],
            aws_access_key_id=config.S3['access_key'],
            aws_secret_access_key=config.S3['secret_key'],
            config=boto_config
        )
    except Exception as e:
        print(f"❌ Failed to initialize S3 client: {e}")
        return None
# ===========================
# INITIALIZATION
# ===========================
# Initialize Supabase recorder
def init_supabase():
    """Initialize Supabase recorder on first use"""
    global supabase_recorder
    if supabase_recorder is None:
        try:
            if config.SUPABASE['enabled']:
                supabase_recorder = SupabaseRecorder(
                    config.SUPABASE['url'],
                    config.SUPABASE['key']
                )
                print("✅ Supabase recorder initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Supabase: {e}")
            raise
    return supabase_recorder
# ===========================
# S3 DEFAULT SIGNATURES LOADER
# ===========================
def load_default_signatures_from_s3():
    """
    ✅ OPTIMIZED: Download default signatures in parallel
    """
    try:
        print("\n" + "="*60)
        print("📥 LOADING DEFAULT SIGNATURES FROM S3")
        print("="*60)
        
        s3 = get_s3_client()
        if not s3:
            raise Exception("S3 client not available")
        
        bucket = config.S3['bucket']
        
        # ✅ STEP 1: List all objects
        print("\n📂 Listing S3 objects...")
        
        unknown_response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix='Default-classes/unknown/'
        )
        
        non_sig_response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix='Default-classes/non-signature/'
        )
        
        unknown_keys = [
            obj['Key'] for obj in unknown_response.get('Contents', [])
            if not obj['Key'].endswith('/') and 
            obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
        
        non_sig_keys = [
            obj['Key'] for obj in non_sig_response.get('Contents', [])
            if not obj['Key'].endswith('/') and 
            obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
        
        print(f"   Found {len(unknown_keys)} unknown images")
        print(f"   Found {len(non_sig_keys)} non-signature images")
        
        # ✅ STEP 2: Download in parallel
        print("\n📥 Downloading in parallel...")
        start_time = time.time()
        
        all_keys = unknown_keys + non_sig_keys
        download_results = download_signature_batch(s3, bucket, all_keys)
        
        elapsed = time.time() - start_time
        print(f"✅ Downloaded {len(download_results)} images in {elapsed:.2f}s")
        
        # ✅ STEP 3: Split into categories
        image_map = {s3_key: img_data for s3_key, img_data in download_results if img_data is not None}
        
        unknown_images = [image_map[key] for key in unknown_keys if key in image_map]
        non_sig_images = [image_map[key] for key in non_sig_keys if key in image_map]
        
        print(f"\n✅ Loaded {len(unknown_images)} unknown, {len(non_sig_images)} non-signatures")
        
        return {
            'unknown': unknown_images,
            'non-signature': non_sig_images
        }
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR loading defaults from S3: {e}")
        import traceback
        traceback.print_exc()
        return None

# ===========================
# CANCEL UPLOAD ENDPOINT
# ===========================
# Global tracking for active uploads
# ✅ ADD LOCKS FOR THREAD SAFETY
active_uploads = {}  # {job_id: {'cancelled': False}}
upload_locks = {}    # {job_id: Lock()} - Prevents race conditions

# ===========================
# BASIC API ENDPOINTS
# ===========================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with S3/Supabase status"""
    s3_configured = config.S3['enabled'] and config.S3['access_key'] and config.S3['secret_key']
    
    return jsonify({
        'status': 'online',
        'tfjs_enabled': config.TFJS['enabled'],
        's3_enabled': config.S3['enabled'],
        's3_configured': s3_configured,
        's3_bucket': config.S3['bucket'] if config.S3['enabled'] else None,
        'supabase_enabled': config.SUPABASE['enabled'],
        'timestamp': datetime.now().isoformat()
    })
@app.route('/config', methods=['GET'])
def get_config():
    """Get backend configuration"""
    return jsonify({
        'model': {
            'input_shape': list(config.MODEL['input_shape']),
            'embedding_dim': config.MODEL['feature_dim'],
            'backbone': config.MODEL['backbone'],
            'dropout': config.MODEL['dropout_rate'],
            'l2_regularization': config.MODEL['l2_regularization']
        },
        'training': {
            'epochs': config.TRAINING['epochs'],
            'batch_size': config.TRAINING['batch_size'],
            'learning_rate': config.TRAINING['learning_rate'],
            'validation_split': config.TRAINING['validation_split'],
            'augmentation_count': config.TRAINING['augmentation_count'],
            'min_samples': config.TRAINING['min_samples_per_class']
        }
    })
@app.route('/api/cancel-upload/<job_id>', methods=['POST'])
def cancel_upload(job_id: str):
    """
    ✅ UPDATED: Cancel with better error handling
    """
    try:
        if not config.SUPABASE['enabled']:
            return jsonify({
                'success': False,
                'message': 'Supabase not enabled'
            }), 503
        
        from supabase import create_client
        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        # ✅ Mark for cancellation FIRST (before checking DB)
        if job_id in active_uploads:
            active_uploads[job_id]['cancelled'] = True
            print(f"🛑 Marked job {job_id} for cancellation in memory")
        
        # Check if job exists in DB
        try:
            response = supabase.table('upload_jobs').select('*').eq('job_id', job_id).single().execute()
            
            if not response.data:
                # Job not in DB yet, but marked for cancellation
                print(f"⚠️ Job {job_id} not in DB yet, but marked for cancellation")
                return jsonify({
                    'success': True,
                    'message': 'Job marked for cancellation'
                })
            
            job = response.data
            
            # Only cancel if job is active
            if job['status'] not in ['pending', 'processing', 'uploading']:
                return jsonify({
                    'success': False,
                    'message': f'Job is already {job["status"]}'
                }), 400
            
            # Update status in Supabase
            supabase.table('upload_jobs').update({
                'status': 'cancelled',
                'completed_at': datetime.now().isoformat()
            }).eq('job_id', job_id).execute()
            
            print(f"🛑 Upload job {job_id} cancelled in DB")
            
        except Exception as db_error:
            # If DB query fails, still return success if we marked it for cancellation
            if job_id in active_uploads and active_uploads[job_id]['cancelled']:
                print(f"⚠️ DB error but job marked for cancellation: {db_error}")
                return jsonify({
                    'success': True,
                    'message': 'Job marked for cancellation (DB pending)'
                })
            raise
        
        return jsonify({
            'success': True,
            'message': 'Upload cancelled successfully'
        })
        
    except Exception as e:
        print(f"❌ Error cancelling upload: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
@app.route('/api/batch-upload-signatures-raw', methods=['POST'])
def batch_upload_signatures_raw():
    """
    ✅ FIXED: Use school_student_id for signature records
    """
    try:
        s3 = get_s3_client()
        if not s3:
            return jsonify({
                'success': False,
                'message': 'S3 service is not available',
                'error': 'S3_NOT_CONFIGURED'
            }), 503
        
        if not config.SUPABASE['enabled']:
            return jsonify({
                'success': False,
                'message': 'Supabase is required for batch uploads'
            }), 503
        
        # Get admin_id from request
        admin_id = request.form.get('admin_id')
        if not admin_id:
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
                try:
                    from supabase import create_client
                    supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
                    user_response = supabase.auth.get_user(token)
                    admin_id = user_response.user.id if user_response.user else None
                except:
                    pass
        
        if not admin_id:
            return jsonify({
                'success': False,
                'message': 'Admin ID is required'
            }), 401
        
        is_first_batch = request.form.get('is_first_batch') == 'true'
        job_id = request.form.get('job_id')
        
        # Extract files from FormData
        file_tasks = []
        file_index = 0
        
        while True:
            file_key = f'file_{file_index}'
            if file_key not in request.files:
                break
            
            file = request.files[file_key]
            school_student_id = request.form.get(f'{file_key}_student_id')  # ✅ This is school ID (e.g., "2023-001")
            student_db_id = request.form.get(f'{file_key}_student_db_id')   # ✅ This is database ID
            file_type = request.form.get(f'{file_key}_type', 'uploaded')
            
            file_buffer = file.read()
            
            file_tasks.append({
                'school_student_id': school_student_id,  # ✅ School ID
                'student_db_id': int(student_db_id),     # ✅ Database ID
                'file_name': file.filename,
                'file_buffer': file_buffer,
                'type': file_type,
                'content_type': file.content_type or 'image/png',
                'admin_id': admin_id
            })
            
            file_index += 1
        
        if not file_tasks:
            return jsonify({
                'success': False,
                'message': 'No files provided'
            }), 400
        
        # First batch: Create job
        if is_first_batch:
            total_files = int(request.form.get('total_files', len(file_tasks)))
            student_count = int(request.form.get('student_count', 1))
            
            from supabase import create_client
            supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
            
            job_response = supabase.table('upload_jobs').insert({
                'total_files': total_files,
                'student_count': student_count,
                'status': 'preparing',
                'uploaded_files': 0,
                'failed_files': 0,
                'admin_id': admin_id
            }).execute()
            
            if not job_response.data:
                raise Exception('Failed to create upload job')
            
            job_record = job_response.data[0]
            job_id = str(job_record['job_id'])
            
            print(f"\n📦 Raw File Batch Upload: {job_id} (Admin: {admin_id})")
            print(f"   Total files: {total_files}")
            print(f"   This batch: {len(file_tasks)}")
            
            active_uploads[job_id] = {'cancelled': False}
            upload_locks[job_id] = Lock()
        else:
            print(f"\n📦 Continuation batch for job: {job_id}")
            print(f"   Files in this batch: {len(file_tasks)}")
            
            if job_id not in upload_locks:
                upload_locks[job_id] = Lock()
        
        # Upload files in background thread
        def upload_batch():
            try:
                bucket = config.S3['bucket']
                max_workers = 100
                
                if active_uploads.get(job_id, {}).get('cancelled', False):
                    return
                
                # Change status to "uploading" when FIRST batch starts
                if is_first_batch:
                    from supabase import create_client
                    supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
                    supabase.table('upload_jobs').update({
                        'status': 'uploading'
                    }).eq('job_id', job_id).execute()
                    print(f"🔄 Status changed to 'uploading' for job {job_id}")
                
                uploaded_count = 0
                failed_count = 0
                first_update_done = False
                
                batch_start = time.time()
                print(f"🚀 Starting upload of {len(file_tasks)} files...")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            upload_single_signature_to_s3,
                            s3,
                            bucket,
                            file_task
                        ): file_task
                        for file_task in file_tasks
                    }
                    
                    processed = 0
                    for future in as_completed(futures):
                        if active_uploads.get(job_id, {}).get('cancelled', False):
                            break
                        
                        file_task = futures[future]
                        processed += 1
                        
                        try:
                            result = future.result()
                            
                            if result['success']:
                                uploaded_count += 1
                                
                                # ✅ CRITICAL FIX: Insert with BOTH student_id (FK) and school_student_id
                                try:
                                    from supabase import create_client
                                    supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
                                    
                                    supabase.table('student_signatures').insert({
                                        'student_id': file_task['student_db_id'],         # ✅ FK to students.id
                                        'school_student_id': file_task['school_student_id'],  # ✅ School ID (e.g., "2023-001")
                                        'label': 'genuine',
                                        's3_key': result['s3_key'],
                                        's3_url': result['s3_url'],
                                        'content_hash': result['content_hash'],
                                        'type': result['type']
                                    }).execute()
                                    
                                except Exception as db_error:
                                    error_str = str(db_error).lower()
                                    
                                    if '23505' in error_str or 'duplicate' in error_str:
                                        if 'student_signatures_school_student_content_hash_key' in error_str:
                                            print(f"⏭️ Skipping duplicate signature for student {file_task['school_student_id']}: {file_task['file_name']}")
                                        else:
                                            print(f"⚠️ Unexpected duplicate: {db_error}")
                                    else:
                                        print(f"⚠️ DB error: {db_error}")
                                        failed_count += 1
                                        uploaded_count -= 1
                            
                            # Update DB after FIRST 30 files, mark as done
                            if is_first_batch and processed == 30 and not first_update_done:
                                with upload_locks[job_id]:
                                    from supabase import create_client
                                    supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
                                    supabase.table('upload_jobs').update({
                                        'uploaded_files': uploaded_count,
                                        'failed_files': failed_count
                                    }).eq('job_id', job_id).execute()
                                first_update_done = True
                                print(f"✅ First 30 files uploaded - UI should update now")
                            
                            # Progress indicator every 20 files
                            if processed % 20 == 0:
                                elapsed = time.time() - batch_start
                                rate = processed / elapsed if elapsed > 0 else 0
                                print(f"   Progress: {processed}/{len(file_tasks)} ({rate:.1f} files/sec)")
                        
                        except Exception as e:
                            failed_count += 1
                            print(f"❌ Upload error: {e}")
                
                # Batch timing
                batch_elapsed = time.time() - batch_start
                batch_rate = len(file_tasks) / batch_elapsed if batch_elapsed > 0 else 0
                print(f"✅ Batch complete: {len(file_tasks)} files in {batch_elapsed:.1f}s ({batch_rate:.1f} files/sec)")
                
                # ATOMIC UPDATE - only add the NEW files since last update
                if uploaded_count > 0 or failed_count > 0:
                    with upload_locks[job_id]:
                        from supabase import create_client
                        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
                        
                        current_job = supabase.table('upload_jobs').select('uploaded_files, failed_files, total_files').eq('job_id', job_id).single().execute()
                        
                        if current_job.data:
                            if is_first_batch and first_update_done:
                                delta_uploaded = uploaded_count - 30
                                delta_failed = failed_count
                                new_uploaded = current_job.data['uploaded_files'] + delta_uploaded
                                new_failed = current_job.data['failed_files'] + delta_failed
                            elif is_first_batch and not first_update_done:
                                new_uploaded = current_job.data['uploaded_files'] + uploaded_count
                                new_failed = current_job.data['failed_files'] + failed_count
                            else:
                                new_uploaded = current_job.data['uploaded_files'] + uploaded_count
                                new_failed = current_job.data['failed_files'] + failed_count
                            
                            total = current_job.data['total_files']
                            
                            supabase.table('upload_jobs').update({
                                'uploaded_files': new_uploaded,
                                'failed_files': new_failed
                            }).eq('job_id', job_id).execute()
                            
                            processed_total = new_uploaded + new_failed
                            print(f"   Job progress: {processed_total}/{total}")
                            
                            # Check if complete
                            if processed_total >= total:
                                supabase.table('upload_jobs').update({
                                    'status': 'completed',
                                    'completed_at': datetime.now().isoformat()
                                }).eq('job_id', job_id).execute()
                                
                                print(f"🎉 Job {job_id} completed! Total: {new_uploaded}/{total}")
                                
                                # Cleanup
                                if job_id in active_uploads:
                                    del active_uploads[job_id]
                                if job_id in upload_locks:
                                    del upload_locks[job_id]
                
            except Exception as e:
                print(f"❌ Batch upload failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Mark job as failed
                try:
                    from supabase import create_client
                    supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
                    supabase.table('upload_jobs').update({
                        'status': 'failed',
                        'error_message': str(e),
                        'completed_at': datetime.now().isoformat()
                    }).eq('job_id', job_id).execute()
                except:
                    pass
        
        # Start upload in background thread
        thread = threading.Thread(target=upload_batch, daemon=True)
        thread.start()
        
        # Return immediately
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Batch received and processing started'
        })
        
    except Exception as e:
        print(f"❌ Endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
@app.route('/api/batch-upload-signatures', methods=['POST'])
def batch_upload_signatures():
    """
    ✅ FIXED: Batch upload with cancellation support
    """
    try:
        s3 = get_s3_client()
        if not s3:
            return jsonify({
                'success': False,
                'message': 'S3 service is not available',
                'error': 'S3_NOT_CONFIGURED'
            }), 503
        
        if not config.SUPABASE['enabled']:
            return jsonify({
                'success': False,
                'message': 'Supabase is required for batch uploads'
            }), 503
        
        data = request.get_json()
        uploads = data.get('uploads', [])
        
        if not uploads:
            return jsonify({
                'success': False,
                'message': 'No uploads provided'
            }), 400
        
        # Count total files
        total_files = sum(len(upload['files']) for upload in uploads)
        total_students = len(uploads)
        
        from supabase import create_client
        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        # ✅ Create job record in Supabase
        job_response = supabase.table('upload_jobs').insert({
            'total_files': total_files,
            'student_count': total_students,
            'status': 'pending'
        }).execute()
        
        if not job_response.data:
            raise Exception('Failed to create upload job')
        
        job_record = job_response.data[0]
        job_id = str(job_record['job_id'])
        
        print(f"\n📦 Batch Upload Job: {job_id}")
        print(f"   Students: {total_students}")
        print(f"   Total files: {total_files}")
        
        # ✅ Register as active upload
        active_uploads[job_id] = {'cancelled': False}
        
        # Prepare file data for workers
        file_tasks = []
        for upload in uploads:
            student_id = upload['student_id']
            student_db_id = upload['student_db_id']
            
            for file_info in upload['files']:
                # Decode base64
                file_data_b64 = file_info['file_data']
                if ',' in file_data_b64:
                    file_data_b64 = file_data_b64.split(',')[1]
                
                file_buffer = base64.b64decode(file_data_b64)
                
                file_tasks.append({
                    'student_id': student_id,
                    'student_db_id': student_db_id,
                    'file_name': file_info['file_name'],
                    'file_buffer': file_buffer,
                    'type': file_info.get('type', 'uploaded'),
                    'content_type': file_info.get('content_type', 'image/png')
                })
        
        # Start background upload
        def upload_background():
            try:
                bucket = config.S3['bucket']
                max_workers = 50
                
                # ✅ Check for cancellation before starting
                if active_uploads.get(job_id, {}).get('cancelled', False):
                    print(f"🛑 Job {job_id} cancelled before upload started")
                    return
                
                # ✅ Update status to 'processing'
                supabase.table('upload_jobs').update({
                    'status': 'processing'
                }).eq('job_id', job_id).execute()
                
                print(f"🚀 Starting S3 upload with {max_workers} workers...")
                
                # ✅ Update status to 'uploading'
                supabase.table('upload_jobs').update({
                    'status': 'uploading'
                }).eq('job_id', job_id).execute()
                
                uploaded_count = 0
                failed_count = 0
                cancelled = False
                
                # Upload in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            upload_single_signature_to_s3,
                            s3,
                            bucket,
                            file_task
                        ): file_task
                        for file_task in file_tasks
                    }
                    
                    # ✅ Check BEFORE starting loop
                    if active_uploads.get(job_id, {}).get('cancelled', False):
                        print(f"🛑 Job {job_id} cancelled before loop started")
                        cancelled = True
                        for f in futures:
                            if not f.done():
                                f.cancel()
                    
                    for future in as_completed(futures):
                        # ✅ Check for cancellation in each iteration
                        if active_uploads.get(job_id, {}).get('cancelled', False):
                            print(f"🛑 Job {job_id} cancelled during upload")
                            cancelled = True
                            
                            # Cancel remaining futures
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            break
                        
                        file_task = futures[future]
                        try:
                            result = future.result()
                            
                            if result['success']:
                                # ✅ Check if skipped
                                if result.get('skipped'):
                                    print(f"⏭️ Skipped duplicate: {result['file_name']}")
                                    uploaded_count += 1
                                    continue
                                
                                uploaded_count += 1
                                
                                # Insert to student_signatures table
                                try:
                                    supabase.table('student_signatures').insert({
                                        'student_id': file_task['student_db_id'],
                                        'label': 'genuine',
                                        's3_key': result['s3_key'],
                                        's3_url': result['s3_url'],
                                        'content_hash': result['content_hash'],
                                        'type': result['type']
                                    }).execute()
                                except Exception as db_error:
                                    print(f"⚠️ Supabase insert failed: {db_error}")
                                    failed_count += 1
                                    uploaded_count -= 1
                            else:
                                failed_count += 1
                        
                        except Exception as e:
                            print(f"❌ Upload failed: {e}")
                            failed_count += 1
                        
                        # ✅ FIXED: Update progress after EVERY file
                        supabase.table('upload_jobs').update({
                            'uploaded_files': uploaded_count,
                            'failed_files': failed_count
                        }).eq('job_id', job_id).execute()
                
                # ✅ Final status update
                if cancelled:
                    supabase.table('upload_jobs').update({
                        'uploaded_files': uploaded_count,
                        'failed_files': failed_count
                    }).eq('job_id', job_id).execute()
                    
                    print(f"\n🛑 Upload cancelled")
                    print(f"   Uploaded: {uploaded_count}/{total_files}")
                    print(f"   Failed: {failed_count}")
                else:
                    # Mark as completed
                    supabase.table('upload_jobs').update({
                        'status': 'completed',
                        'uploaded_files': uploaded_count,
                        'failed_files': failed_count,
                        'completed_at': datetime.now().isoformat()
                    }).eq('job_id', job_id).execute()
                    
                    print(f"\n✅ Batch upload completed")
                    print(f"   Uploaded: {uploaded_count}/{total_files}")
                    print(f"   Failed: {failed_count}")
                
            except Exception as e:
                print(f"❌ Batch upload failed: {e}")
                import traceback
                traceback.print_exc()
                
                # ✅ Mark as failed
                supabase.table('upload_jobs').update({
                    'status': 'failed',
                    'error_message': str(e),
                    'completed_at': datetime.now().isoformat()
                }).eq('job_id', job_id).execute()
            
            finally:
                # ✅ Clean up active upload tracking
                if job_id in active_uploads:
                    del active_uploads[job_id]
        
        # Start background thread
        thread = threading.Thread(target=upload_background, daemon=True)
        thread.start()
        
        # ✅ Return immediately
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Batch upload started',
            'total_files': total_files,
            'students': total_students
        })
        
    except Exception as e:
        print(f"❌ Batch upload endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
def upload_single_signature_to_s3(s3_client, bucket: str, file_data: Dict) -> Dict:
    """
    ✅ FIXED: Upload signature using school_student_id
    """
    school_student_id = file_data['school_student_id']  # ✅ School ID (e.g., "2023-001")
    file_name = file_data['file_name']
    admin_id = file_data.get('admin_id')
    
    try:
        content_hash = hashlib.sha256(file_data['file_buffer']).hexdigest()
        
        timestamp = int(time.time() * 1000000)
        safe_filename = ''.join(c for c in file_name if c.isalnum() or c in '._-')
        subfolder = 'Uploaded' if file_data['type'] == 'uploaded' else 'Captured'
        
        # ✅ Use school_student_id in S3 path
        if admin_id:
            s3_key = f"Admins/{admin_id}/Students/{school_student_id}/Signatures/{subfolder}/{timestamp}_{safe_filename}"
        else:
            s3_key = f"Students/{school_student_id}/Signatures/{subfolder}/{timestamp}_{safe_filename}"
        
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=file_data['file_buffer'],
            ContentType=file_data.get('content_type', 'image/png'),
            Metadata={
                'upload-type': file_data['type'],
                'school-student-id': str(school_student_id),  # ✅ School ID in metadata
                'admin-id': str(admin_id) if admin_id else ''
            }
        )
        
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': s3_key},
            ExpiresIn=604800
        )
        
        return {
            'success': True,
            'school_student_id': school_student_id,  # ✅ Return school ID
            'file_name': file_name,
            's3_key': s3_key,
            's3_url': presigned_url,
            'content_hash': content_hash,
            'type': file_data['type']
        }
        
    except Exception as e:
        return {
            'success': False,
            'school_student_id': school_student_id,
            'file_name': file_name,
            'error': str(e)
        }
@app.route('/api/batch-upload-status/<job_id>', methods=['GET'])
def get_batch_upload_status(job_id: str):
    """Get batch upload job status"""
    try:
        if not config.SUPABASE['enabled']:
            return jsonify({
                'success': False,
                'message': 'Supabase not enabled'
            }), 503
        
        from supabase import create_client
        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        # ✅ Get current user
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                'success': False,
                'message': 'Unauthorized'
            }), 401
        
        token = auth_header.split(' ')[1]
        user_response = supabase.auth.get_user(token)
        if not user_response.user:
            return jsonify({
                'success': False,
                'message': 'Unauthorized'
            }), 401
        
        admin_id = user_response.user.id
        
        # ✅ FIX: Use maybe_single() and handle None
        response = supabase.table('upload_jobs')\
            .select('*')\
            .eq('job_id', job_id)\
            .eq('admin_id', admin_id)\
            .maybe_single()\
            .execute()
        
        # ✅ Check if job exists
        if not response or not response.data:
            return jsonify({
                'success': False,
                'message': 'Job not found or access denied'
            }), 404
        
        return jsonify({
            'success': True,
            'job': response.data
        })
        
    except Exception as e:
        print(f"❌ Error getting upload status: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/active-upload-jobs', methods=['GET'])
def get_active_upload_jobs():
    """Get all active (incomplete) upload jobs"""
    try:
        if not config.SUPABASE['enabled']:
            return jsonify({
                'success': False,
                'message': 'Supabase not enabled'
            }), 503
        
        from supabase import create_client
        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        # ✅ FIX: Handle empty result
        response = supabase.table('upload_jobs')\
            .select('*')\
            .in_('status', ['pending', 'processing', 'uploading'])\
            .order('created_at', desc=True)\
            .execute()
        
        # ✅ Return empty array if no jobs
        return jsonify({
            'success': True,
            'jobs': response.data if response and response.data else []
        })
        
    except Exception as e:
        print(f"❌ Error getting active jobs: {e}")
        return jsonify({
            'success': True,  # ✅ Don't fail
            'jobs': []
        })

@app.route('/api/batch-upload-cleanup/<job_id>', methods=['DELETE'])
def cleanup_batch_upload_job(job_id: str):
    """Cleanup is handled by Supabase - this is a no-op"""
    return jsonify({'success': True, 'message': 'Job tracking in Supabase'})
# ===========================
# ✅ NEW: S3 FILE OPERATIONS
# ===========================
@app.route('/api/upload-signature', methods=['POST'])
def upload_signature():
    """
    ✅ PYTHON VERSION: Upload signature to S3
    Replaces Node.js POST /api/upload-signature
    """
    try:
        s3 = get_s3_client()
        if not s3:
            return jsonify({
                'success': False,
                'message': 'S3 service is not available. Please configure AWS credentials.',
                'error': 'S3_NOT_CONFIGURED'
            }), 503
        
        # Get form data
        student_id = request.form.get('studentId')
        signature_type = request.form.get('type', 'uploaded')
        signature_file = request.files.get('signature')
        
        print(f'📥 Received upload request:')
        print(f'   Student ID: {student_id}')
        print(f'   Type: {signature_type}')
        print(f'   File: {signature_file.filename if signature_file else "No file"}')
        
        if not student_id or not signature_file:
            return jsonify({
                'success': False,
                'message': 'Missing required fields: studentId and signature file'
            }), 400
        
        # Validate type
        if signature_type not in ['uploaded', 'captured']:
            return jsonify({
                'success': False,
                'message': 'Invalid type. Must be "uploaded" or "captured"'
            }), 400
        
        # Generate S3 key
        timestamp = int(time.time() * 1000)
        filename = signature_file.filename.replace(' ', '_')
        safe_filename = ''.join(c for c in filename if c.isalnum() or c in '._-')
        
        subfolder = 'Uploaded' if signature_type == 'uploaded' else 'Captured'
        s3_key = f"Students/{student_id}/Signatures/{subfolder}/{timestamp}_{safe_filename}"
        
        print(f'📤 Uploading to S3:')
        print(f'   Bucket: {config.S3["bucket"]}')
        print(f'   Key: {s3_key}')
        print(f'   Subfolder: {subfolder}')
        
        # Upload to S3
        s3.put_object(
            Bucket=config.S3['bucket'],
            Key=s3_key,
            Body=signature_file.read(),
            ContentType=signature_file.content_type or 'image/png',
            Metadata={
                'upload-type': signature_type,
                'student-id': str(student_id)
            }
        )
        
        # Generate presigned URL (valid for 7 days)
        presigned_url = s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': config.S3['bucket'],
                'Key': s3_key
            },
            ExpiresIn=604800  # 7 days
        )
        
        public_url = f"https://{config.S3['bucket']}.s3.{config.S3['region']}.amazonaws.com/{s3_key}"
        
        print(f'✅ Uploaded successfully')
        print(f'   Final S3 Key: {s3_key}')
        
        return jsonify({
            'success': True,
            's3_key': s3_key,
            's3_url': presigned_url,
            'presigned_url': presigned_url,
            'public_url': public_url,
            'student_id': student_id,
            'type': signature_type
        })
        
    except Exception as e:
        print(f'❌ Error uploading signature: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
@app.route('/api/generate-signature-url', methods=['POST'])
def generate_signature_url():
    """
    ✅ PYTHON VERSION: Generate presigned URL for single signature
    Replaces Node.js POST /api/generate-signature-url
    """
    try:
        s3 = get_s3_client()
        if not s3:
            return jsonify({
                'success': False,
                'message': 'S3 service is not available',
                'error': 'S3_NOT_CONFIGURED'
            }), 503
        
        data = request.get_json()
        s3_key = data.get('s3_key')
        
        if not s3_key:
            return jsonify({
                'success': False,
                'message': 'Missing s3_key parameter'
            }), 400
        
        # Generate presigned URL (valid for 7 days)
        print(f'Generating presigned URL for: {s3_key}')
        presigned_url = s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': config.S3['bucket'],
                'Key': s3_key
            },
            ExpiresIn=604800  # 7 days
        )
        
        print(f'✅ Generated presigned URL: {presigned_url[:80]}...')
        
        return jsonify({
            'success': True,
            'url': presigned_url
        })
        
    except Exception as e:
        print(f'❌ Error generating presigned URL: {e}')
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
@app.route('/api/generate-signature-urls-batch', methods=['POST'])
def generate_signature_urls_batch():
    """
    ✅ PYTHON VERSION: Generate presigned URLs in batch
    Replaces Node.js POST /api/generate-signature-urls-batch
    """
    try:
        s3 = get_s3_client()
        if not s3:
            return jsonify({
                'success': False,
                'message': 'S3 service is not available',
                'error': 'S3_NOT_CONFIGURED'
            }), 503
        
        data = request.get_json()
        s3_keys = data.get('s3_keys', [])
        
        if not isinstance(s3_keys, list) or len(s3_keys) == 0:
            return jsonify({
                'success': False,
                'message': 'Missing or invalid s3_keys parameter (must be array)'
            }), 400
        
        print(f'📦 Generating {len(s3_keys)} presigned URLs in batch...')
        
        def generate_url(s3_key):
            """Generate presigned URL for a single key"""
            try:
                presigned_url = s3.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': config.S3['bucket'],
                        'Key': s3_key
                    },
                    ExpiresIn=604800  # 7 days
                )
                return {
                    's3_key': s3_key,
                    'url': presigned_url,
                    'success': True
                }
            except Exception as e:
                print(f'❌ Failed to generate URL for {s3_key}: {e}')
                return {
                    's3_key': s3_key,
                    'url': None,
                    'success': False,
                    'error': str(e)
                }
        
        # Generate all URLs in parallel
        with ThreadPoolExecutor(max_workers=50) as executor:
            results = list(executor.map(generate_url, s3_keys))
        
        successful = len([r for r in results if r['success']])
        print(f'✅ Generated {successful}/{len(s3_keys)} URLs')
        
        return jsonify({
            'success': True,
            'urls': results
        })
        
    except Exception as e:
        print(f'❌ Error generating presigned URLs: {e}')
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
@app.route('/api/delete-signature', methods=['DELETE'])
def delete_signature():
    """
    ✅ PYTHON VERSION: Delete signature from S3
    Replaces Node.js DELETE /api/delete-signature
    """
    try:
        s3 = get_s3_client()
        if not s3:
            return jsonify({
                'success': False,
                'message': 'S3 service is not available',
                'error': 'S3_NOT_CONFIGURED'
            }), 503
        
        data = request.get_json()
        s3_key = data.get('s3_key')
        
        if not s3_key:
            return jsonify({
                'success': False,
                'message': 'Missing s3_key parameter'
            }), 400
        
        # Delete from S3
        print(f'🗑️ Deleting from S3: {s3_key}')
        s3.delete_object(
            Bucket=config.S3['bucket'],
            Key=s3_key
        )
        
        print(f'✅ Deleted successfully')
        
        return jsonify({
            'success': True,
            'message': 'Signature deleted successfully'
        })
        
    except Exception as e:
        print(f'❌ Error deleting signature: {e}')
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
@app.route('/api/upload-model-to-s3', methods=['POST'])
def upload_model_to_s3():
    """
    ✅ PYTHON VERSION: Upload model to S3 (3-file format)
    Replaces Node.js POST /api/upload-model-to-s3
    """
    try:
        s3 = get_s3_client()
        if not s3:
            return jsonify({
                'success': False,
                'message': 'S3 service is not available. Please configure AWS credentials.',
                'error': 'S3_NOT_CONFIGURED'
            }), 503
        
        data = request.get_json()
        model_data = data.get('modelData')
        metadata = data.get('metadata')
        is_three_file_format = data.get('isThreeFileFormat', False)
        
        if not model_data:
            return jsonify({
                'success': False,
                'message': 'Missing modelData'
            }), 400
        
        # Generate timestamp and folder structure
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder_path = f"ai-models/{timestamp}"
        
        if is_three_file_format:
            # NEW FORMAT: 3 files (model.json, weights.bin, metadata.json)
            print(f'📦 Uploading 3-file model to {folder_path}')
            
            # 1. Upload model.json
            model_json_key = f"{folder_path}/model.json"
            s3.put_object(
                Bucket=config.S3['bucket'],
                Key=model_json_key,
                Body=model_data['modelJson'].encode('utf-8'),
                ContentType='application/json'
            )
            print(f'✅ Uploaded model.json')
            
            # 2. Upload weights.bin (decode base64)
            weights_base64 = model_data['weightsBin']
            if ',' in weights_base64:
                weights_base64 = weights_base64.split(',')[1]
            
            weights_buffer = base64.b64decode(weights_base64)
            weights_key = f"{folder_path}/weights.bin"
            
            s3.put_object(
                Bucket=config.S3['bucket'],
                Key=weights_key,
                Body=weights_buffer,
                ContentType='application/octet-stream'
            )
            print(f'✅ Uploaded weights.bin ({len(weights_buffer)} bytes)')
            
            # 3. Upload metadata.json
            metadata_key = f"{folder_path}/metadata.json"
            s3.put_object(
                Bucket=config.S3['bucket'],
                Key=metadata_key,
                Body=model_data['metadataJson'].encode('utf-8'),
                ContentType='application/json'
            )
            print(f'✅ Uploaded metadata.json')
            
            # Return success
            model_url = f"https://{config.S3['bucket']}.s3.{config.S3['region']}.amazonaws.com/{model_json_key}"
            
            return jsonify({
                'success': True,
                'location': model_url,
                'metadata': {
                    'storage': {
                        'location': 's3',
                        'bucket': config.S3['bucket'],
                        'region': config.S3['region'],
                        'modelKey': model_json_key,
                        'weightsKey': weights_key,
                        'metadataKey': metadata_key
                    }
                },
                'message': 'Model uploaded successfully (3-file format)'
            })
        else:
            # OLD FORMAT: Not supported
            print('⚠️ Old format upload attempt rejected')
            return jsonify({
                'success': False,
                'message': '5-file format is deprecated. Please use 3-file format (isThreeFileFormat=true)'
            }), 400
            
    except Exception as e:
        print(f'❌ Error uploading model: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
# filepath: python/main.py (ADDITIONAL FIX - Download Model Endpoint)

@app.route('/api/download-model/<model_uuid>', methods=['GET'])
def download_model_from_s3(model_uuid):
    """
    ✅ FIXED: Download model from S3 with better error handling
    """
    try:
        print(f'🔄 Downloading model: {model_uuid}')
        
        # ✅ Initialize S3 client FIRST
        s3 = get_s3_client()
        if not s3:
            return jsonify({
                'success': False,
                'message': 'S3 service is not available',
                'error': 'S3_NOT_CONFIGURED'
            }), 503
        
        # ✅ Initialize Supabase client
        from supabase import create_client
        supabase_client = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        # ✅ Get admin_id from Authorization header
        admin_id = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            try:
                user_response = supabase_client.auth.get_user(token)
                if user_response.user:
                    admin_id = user_response.user.id
                    print(f'✅ Admin authenticated: {admin_id}')
            except Exception as e:
                print(f'⚠️ Failed to verify token: {e}')
        
        # Allow fallback for now, but log warning
        if not admin_id:
            print(f'⚠️ No admin_id provided, attempting public model download')
        
        # ✅ Query model with optional admin filter
        query = supabase_client.table('global_trained_models')\
            .select('*')\
            .eq('model_uuid', model_uuid)
        
        # Apply admin filter if authenticated
        if admin_id:
            admin_response = supabase_client.table('admin')\
                .select('id')\
                .eq('id', admin_id)\
                .maybe_single()\
                .execute()
            
            effective_admin_id = admin_id
            
            if admin_response is None or not admin_response.data:
                # User is attendance_checker, get their admin_id
                checker_response = supabase_client.table('attendance_checker')\
                    .select('admin_id')\
                    .eq('id', admin_id)\
                    .maybe_single()\
                    .execute()
    
                if checker_response is None or not checker_response.data or not checker_response.data.get('admin_id'):
                    print(f'❌ Attendance checker has no linked admin')
                    return jsonify({
                        'success': False,
                        'message': 'Attendance checker has no linked admin'
                    }), 403
    
                effective_admin_id = checker_response.data['admin_id']
                print(f'📋 Attendance checker using admin models: {effective_admin_id}')
            else:
                print(f'👤 Admin logged in: {admin_id}')
            
            # Filter by admin
            query = query.eq('admin_id', effective_admin_id)
        
        # Execute query
        model_response = query.maybe_single().execute()
        
        # ✅ FIX: Better empty database handling
        if not model_response or not model_response.data:
            print(f'❌ Model not found: {model_uuid}')
            if admin_id:
                print(f'   Admin filter: {effective_admin_id}')
            return jsonify({
                'success': False,
                'message': 'Model not found or access denied. Make sure you have trained at least one model.'
            }), 404
        
        model_data = model_response.data
        print(f'✅ Model found: {model_uuid}')
        
        # Determine S3 path
        s3_base_path = ''
        
        if model_data.get('s3_key'):
            s3_key_parts = model_data['s3_key'].split('/')
            if len(s3_key_parts) >= 3:
                s3_base_path = '/'.join(s3_key_parts[:-1])
            else:
                s3_base_path = f"models/classifier/{model_uuid}"
        else:
            s3_base_path = f"models/classifier/{model_uuid}"
        
        print(f'📁 S3 base path: {s3_base_path}')
        
        # Download files from S3
        bucket = config.S3['bucket']
        
        # Step 1: Download model.json
        model_json_key = f"{s3_base_path}/model.json"
        print(f'📥 Downloading: {model_json_key}')
        
        try:
            model_s3_response = s3.get_object(Bucket=bucket, Key=model_json_key)
            model_json_text = model_s3_response['Body'].read().decode('utf-8')
            model_json = json.loads(model_json_text)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                print(f'❌ Model file not found in S3: {model_json_key}')
                return jsonify({
                    'success': False,
                    'message': 'Model files not found in S3. The model may have been deleted or corrupted.'
                }), 404
            raise
        
        # Step 2: Extract weight shard filenames
        weight_shard_files = []
        if 'weightsManifest' in model_json and isinstance(model_json['weightsManifest'], list):
            for manifest in model_json['weightsManifest']:
                if 'paths' in manifest and isinstance(manifest['paths'], list):
                    weight_shard_files.extend(manifest['paths'])
        
        if not weight_shard_files:
            print('⚠️ No weight shards in manifest, trying weights.bin...')
            weight_shard_files = ['weights.bin']
        
        print(f'📦 Downloading {len(weight_shard_files)} weight shard(s)...')
        
        # Step 3: Download all weight shards
        weight_buffers = []
        for shard_filename in weight_shard_files:
            shard_key = f"{s3_base_path}/{shard_filename}"
            print(f'📥 Downloading: {shard_key}')
            
            try:
                shard_response = s3.get_object(Bucket=bucket, Key=shard_key)
                shard_buffer = shard_response['Body'].read()
                weight_buffers.append(shard_buffer)
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'NoSuchKey':
                    print(f'❌ Weight file not found in S3: {shard_key}')
                    return jsonify({
                        'success': False,
                        'message': f'Weight file missing: {shard_filename}'
                    }), 404
                raise
        
        # Combine all weight shards
        combined_weights = b''.join(weight_buffers)
        weights_base64 = base64.b64encode(combined_weights).decode('utf-8')
        
        # Step 4: Download metadata.json
        metadata_key = f"{s3_base_path}/metadata.json"
        print(f'📥 Downloading: {metadata_key}')
        
        try:
            metadata_response = s3.get_object(Bucket=bucket, Key=metadata_key)
            metadata_json = metadata_response['Body'].read().decode('utf-8')
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                print(f'⚠️ Metadata file not found, creating default...')
                # Create minimal metadata
                metadata_json = json.dumps({
                    'labels': [],
                    'created_at': model_data.get('created_at', datetime.now().isoformat())
                })
        
        # Step 5: Update model.json to reference weights.bin
        if 'weightsManifest' in model_json and isinstance(model_json['weightsManifest'], list):
            all_weights = []
            for manifest in model_json['weightsManifest']:
                if 'weights' in manifest:
                    all_weights.extend(manifest['weights'])
            
            model_json['weightsManifest'] = [{
                'paths': ['weights.bin'],
                'weights': all_weights
            }]
        
        # Combine all data
        combined_data = {
            'modelJson': json.dumps(model_json),
            'weightsBin': weights_base64,
            'metadataJson': metadata_json
        }
        
        print(f'✅ Successfully downloaded model')
        
        return jsonify({
            'success': True,
            'data': json.dumps(combined_data),
            'message': 'Model downloaded successfully (3-file format)'
        })
        
    except ClientError as s3_error:
        print(f'❌ S3 error downloading model: {s3_error}')
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Failed to download from S3: {str(s3_error)}'
        }), 500
    except Exception as e:
        print(f'❌ Error downloading model: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
# ===========================
# TRAINING SCHEDULE ENDPOINTS
# ===========================
@app.route('/api/training-settings', methods=['GET'])
def get_training_settings():
    """Get training settings for current admin"""
    try:
        admin_id = get_current_admin_id()
        if not admin_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        from supabase import create_client
        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        # ✅ FIX: Handle None response properly
        try:
            response = supabase.table('training_settings')\
                .select('*')\
                .eq('admin_id', admin_id)\
                .maybe_single()\
                .execute()
            
            # ✅ Check if response exists and has data
            if response and response.data:
                return jsonify({
                    'success': True,
                    'settings': response.data
                })
        except Exception as query_error:
            # If error is because no record exists, create default
            error_str = str(query_error).lower()
            if 'pgrst116' not in error_str and 'no rows' not in error_str:
                # If it's a real error (not "no rows"), raise it
                raise
        
        # ✅ Create default settings if not exists
        default_settings = {
            'admin_id': admin_id,
            'auto_training_enabled': False,
            'default_countdown_minutes': 10,
            'min_new_signatures_per_student': 5
        }
        
        insert_response = supabase.table('training_settings')\
            .insert(default_settings)\
            .execute()
        
        # ✅ Handle insert response properly
        if insert_response and insert_response.data:
            return jsonify({
                'success': True,
                'settings': insert_response.data[0]
            })
        else:
            # If insert fails, return defaults anyway
            return jsonify({
                'success': True,
                'settings': default_settings
            })
        
    except Exception as e:
        print(f"❌ Error getting training settings: {e}")
        import traceback
        traceback.print_exc()
        
        # ✅ Return default settings on error
        return jsonify({
            'success': True,
            'settings': {
                'admin_id': get_current_admin_id(),
                'auto_training_enabled': False,
                'default_countdown_minutes': 10,
                'min_new_signatures_per_student': 5
            }
        })


@app.route('/api/training-settings', methods=['PUT'])
def update_training_settings():
    """Update training settings for current admin"""
    try:
        admin_id = get_current_admin_id()
        if not admin_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        data = request.json
        
        from supabase import create_client
        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        update_data = {}
        if 'default_countdown_minutes' in data:
            update_data['default_countdown_minutes'] = data['default_countdown_minutes']
        if 'min_new_signatures_per_student' in data:
            update_data['min_new_signatures_per_student'] = data['min_new_signatures_per_student']
        if 'auto_training_enabled' in data:
            update_data['auto_training_enabled'] = data['auto_training_enabled']
        
        update_data['updated_at'] = datetime.now().isoformat()
        
        response = supabase.table('training_settings')\
            .update(update_data)\
            .eq('admin_id', admin_id)\
            .execute()
        
        return jsonify({
            'success': True,
            'settings': response.data[0]
        })
        
    except Exception as e:
        print(f"❌ Error updating training settings: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/queue-retrain-check', methods=['POST'])
def queue_retrain_check():
    """
    Queue a retrain check to be executed after current training completes.
    Called by frontend when changes are detected during active training.
    """
    try:
        admin_id = get_current_admin_id()
        if not admin_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        print(f'📝 Retrain check requested for admin: {admin_id}')  # ✅ Use print
        
        from supabase import create_client
        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        # Check if training is active
        active_jobs = supabase.table('training_jobs')\
            .select('job_id, status, pending_retrain_check')\
            .eq('admin_id', admin_id)\
            .eq('status', 'training')\
            .execute()
        
        if not active_jobs.data or len(active_jobs.data) == 0:
            print('⚠️ No active training found - cannot queue retrain check')
            return jsonify({
                'success': False, 
                'message': 'No active training'
            }), 400
        
        job = active_jobs.data[0]
        job_id = job['job_id']
        
        # Check if already queued
        if job.get('pending_retrain_check'):
            print(f'✅ Retrain check already queued for job: {job_id}')
            return jsonify({
                'success': True,
                'message': 'Retrain check already queued',
                'already_queued': True
            })
        
        # Mark for retrain check
        supabase.table('training_jobs').update({
            'pending_retrain_check': True
        }).eq('job_id', job_id).execute()
        
        print(f'✅ Queued retrain check for job: {job_id}')
        
        return jsonify({
            'success': True,
            'message': 'Queued for retrain check',
            'job_id': job_id
        })
        
    except Exception as e:
        print(f'❌ Error queueing retrain check: {e}')  # ✅ Use print
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
# ===========================
# COUNTDOWN API ENDPOINTS
# ===========================

@app.route('/api/training-schedule', methods=['GET'])
def get_training_schedule():
    """Get current countdown status for admin"""
    try:
        admin_id = get_current_admin_id()
        if not admin_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        from supabase import create_client
        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        # Check if training is active
        active_jobs = supabase.table('training_jobs')\
            .select('job_id, status')\
            .eq('admin_id', admin_id)\
            .eq('status', 'training')\
            .execute()
        
        if active_jobs.data and len(active_jobs.data) > 0:
            return jsonify({
                'success': True,
                'schedule': None,
                'reason': 'training_active'
            })
        
        # Get countdown from database
        schedule_response = supabase.table('training_schedule')\
            .select('*')\
            .eq('admin_id', admin_id)\
            .in_('status', ['counting', 'paused'])\
            .order('created_at', desc=True)\
            .limit(1)\
            .maybe_single()\
            .execute()  # ✅ CHANGED: maybeSingle() → maybe_single()
        
        if not schedule_response or not schedule_response.data:
            return jsonify({
                'success': True,
                'schedule': None
            })
        
        schedule = schedule_response.data
        
        # ✅ CRITICAL: Calculate remaining seconds
        if schedule['status'] == 'counting' and schedule['countdown_ends_at']:
            now = datetime.now(timezone.utc)
            ends_at = datetime.fromisoformat(schedule['countdown_ends_at'].replace('Z', '+00:00'))
            
            remaining_seconds = int((ends_at - now).total_seconds())
            remaining_seconds = max(0, remaining_seconds)  # Don't go negative
            
            print(f'⏱️  Countdown remaining: {remaining_seconds}s for admin {admin_id}')
            
            schedule['remaining_seconds'] = remaining_seconds
        else:
            schedule['remaining_seconds'] = 0
        
        return jsonify({
            'success': True,
            'schedule': schedule
        })
        
    except Exception as e:
        print(f'❌ Error getting schedule: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': True,
            'schedule': None
        })

@app.route('/api/training-schedule/start', methods=['POST'])
def start_training_countdown():
    """Start a countdown"""
    try:
        admin_id = get_current_admin_id()
        if not admin_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        data = request.json
        countdown_minutes = float(data.get('countdown_minutes', 1))
        countdown_seconds = int(countdown_minutes * 60)
        
        # ✅ CRITICAL FIX: Read trigger_reason from request body
        trigger_reason = data.get('trigger_reason', 'manual')
        
        print(f"📋 Countdown request received:")
        print(f"   Admin: {admin_id}")
        print(f"   Duration: {countdown_minutes} minutes")
        print(f"   Trigger reason from request: '{trigger_reason}'")  # ✅ Debug
        print(f"   Full request data: {data}")  # ✅ Debug
        
        # Get current training data snapshot
        students = get_students_with_signatures(admin_id)
        total_signatures = sum(s['signature_count'] for s in students)
        
        # ✅ Start countdown with trigger_reason from request
        timer = countdown_manager.start_countdown(
            admin_id=admin_id,
            duration_seconds=countdown_seconds,
            student_count=len(students),
            signature_count=total_signatures,
            trigger_reason=trigger_reason  # ✅ Use the value from request
        )
        
        print(f"✅ Timer created with trigger_reason: '{timer.trigger_reason}'")  # ✅ Debug
        
        return jsonify({
            'success': True,
            'schedule': timer.get_state(),
            'message': f'Countdown started: {countdown_minutes} minutes'
        })
        
    except Exception as e:
        print(f"❌ Error starting countdown: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training-schedule/pause', methods=['POST'])
def pause_training_countdown():
    """Pause countdown"""
    try:
        admin_id = get_current_admin_id()
        if not admin_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        success = countdown_manager.pause_countdown(admin_id)
        
        if not success:
            return jsonify({'success': False, 'error': 'No active countdown'}), 404
        
        timer = countdown_manager.get_countdown(admin_id)
        return jsonify({
            'success': True,
            'schedule': timer.get_state() if timer else None
        })
        
    except Exception as e:
        print(f"❌ Error pausing countdown: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training-schedule/resume', methods=['POST'])
def resume_training_countdown():
    """Resume countdown"""
    try:
        admin_id = get_current_admin_id()
        if not admin_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        success = countdown_manager.resume_countdown(admin_id)
        
        if not success:
            return jsonify({'success': False, 'error': 'No paused countdown'}), 404
        
        timer = countdown_manager.get_countdown(admin_id)
        return jsonify({
            'success': True,
            'schedule': timer.get_state() if timer else None
        })
        
    except Exception as e:
        print(f"❌ Error resuming countdown: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training-schedule/adjust', methods=['POST'])
def adjust_training_countdown():
    """Adjust countdown time"""
    try:
        admin_id = get_current_admin_id()
        if not admin_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        data = request.json
        adjustment_minutes = float(data.get('adjustment_minutes', 0))
        adjustment_seconds = int(adjustment_minutes * 60)
        
        success = countdown_manager.adjust_countdown(admin_id, adjustment_seconds)
        
        if not success:
            return jsonify({'success': False, 'error': 'No active countdown'}), 404
        
        timer = countdown_manager.get_countdown(admin_id)
        return jsonify({
            'success': True,
            'schedule': timer.get_state() if timer else None,
            'message': f'Adjusted by {adjustment_minutes} minutes'
        })
        
    except Exception as e:
        print(f"❌ Error adjusting countdown: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training-schedule/cancel', methods=['POST'])
def cancel_training_countdown():
    """Cancel countdown"""
    try:
        admin_id = get_current_admin_id()
        if not admin_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        success = countdown_manager.cancel_countdown(admin_id)
        
        if not success:
            return jsonify({'success': False, 'error': 'No active countdown'}), 404
        
        return jsonify({
            'success': True,
            'message': 'Countdown cancelled'
        })
        
    except Exception as e:
        print(f"❌ Error cancelling countdown: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training-schedule/start-now', methods=['POST'])
def start_training_now():
    """Complete countdown immediately and start training"""
    try:
        admin_id = get_current_admin_id()
        if not admin_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        success = countdown_manager.complete_countdown(admin_id)
        
        if not success:
            # No countdown active, just start training
            students = get_students_with_signatures(admin_id)
            if len(students) < 2:
                return jsonify({
                    'success': False,
                    'error': 'Need at least 2 students to train'
                }), 400
        
        return jsonify({
            'success': True,
            'message': 'Training started immediately'
        })
        
    except Exception as e:
        print(f"❌ Error starting training now: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def get_current_admin_id():
    """
    ✅ FIXED: Get current admin ID from Authorization header
    Resolves attendance_checker → admin_id if needed
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.split(' ')[1]
    try:
        from supabase import create_client
        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        # Get user from token
        user_response = supabase.auth.get_user(token)
        if not user_response.user:
            return None
        
        user_id = user_response.user.id
        
        # ✅ Step 1: Check if user is an admin
        admin_check = supabase.table('admin')\
            .select('id')\
            .eq('id', user_id)\
            .maybe_single()\
            .execute()
        
        # ✅ FIX: Check if response exists AND has data
        if admin_check and admin_check.data:
            # User is admin, return their ID
            print(f"✅ User is admin: {user_id}")
            return user_id
        
        # ✅ Step 2: Check if user is attendance_checker
        checker_check = supabase.table('attendance_checker')\
            .select('admin_id')\
            .eq('id', user_id)\
            .maybe_single()\
            .execute()
        
        # ✅ FIX: Check if response exists AND has data
        if checker_check and checker_check.data and checker_check.data.get('admin_id'):
            # User is checker, return their admin_id
            admin_id = checker_check.data['admin_id']
            print(f"✅ User is checker, resolved admin: {admin_id}")
            return admin_id
        
        # ✅ Step 3: User is neither admin nor checker
        print(f"⚠️ User {user_id} is neither admin nor attendance_checker")
        return None
        
    except Exception as e:
        print(f"❌ Error resolving admin ID: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_students_with_signatures(admin_id):
    """
    Get students with signature counts for admin
    
    Args:
        admin_id: The RESOLVED admin_id (already checked for checker → admin)
    """
    from supabase import create_client
    supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
    
    # ✅ No need to resolve again - admin_id is already correct
    print(f"📊 Fetching students for admin: {admin_id}")
    
    students_response = supabase.table('students')\
        .select('id, student_id')\
        .eq('admin_id', admin_id)\
        .execute()
    
    if not students_response or not students_response.data:
        print(f"⚠️ No students found for admin {admin_id}")
        return []
    
    students = []
    for student in students_response.data:
        try:
            sig_response = supabase.table('student_signatures')\
                .select('id', count='exact')\
                .eq('student_id', student['id'])\
                .execute()
            
            sig_count = sig_response.count if sig_response and sig_response.count is not None else 0
            
            if sig_count >= 3:
                students.append({
                    'student_id': student['student_id'],
                    'db_id': student['id'],
                    'signature_count': sig_count
                })
        except Exception as e:
            print(f"⚠️ Error getting signatures for student {student['student_id']}: {e}")
            continue
    
    print(f"✅ Found {len(students)} eligible students for admin {admin_id}")
    return students
@app.route('/api/training-schedule/complete', methods=['POST'])
def complete_training_countdown():
    """Mark countdown as completed (finished naturally, not cancelled)"""
    try:
        admin_id = get_current_admin_id()
        if not admin_id:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        from supabase import create_client
        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        response = supabase.table('training_schedule')\
            .update({
                'status': 'completed',
                'updated_at': datetime.now(timezone.utc).isoformat()
            })\
            .eq('admin_id', admin_id)\
            .in_('status', ['counting', 'paused'])\
            .execute()
        
        print(f"✅ Countdown completed for admin {admin_id}")
        
        return jsonify({
            'success': True,
            'message': 'Countdown completed'
        })
        
    except Exception as e:
        print(f"❌ Error completing countdown: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
# ===========================
# PRODUCTION TRAINING ENDPOINT
# ===========================
@app.route('/train_production', methods=['POST'])
def train_production():
    """Production training endpoint with auto-countdown cancellation"""
    try:
        data = request.json
        student_ids = data.get('student_ids', [])
        
        if not student_ids or len(student_ids) < 2:
            return jsonify({
                'success': False,
                'error': 'Need at least 2 students to train'
            }), 400
        
        admin_id = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            try:
                from supabase import create_client
                supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
                user_response = supabase.auth.get_user(token)
                admin_id = user_response.user.id if user_response.user else None
            except:
                pass
        
        if not admin_id:
            return jsonify({'success': False, 'error': 'Admin ID required'}), 401
        
        # Check for recent duplicate requests
        with training_request_lock:
            now = time.time()
            if admin_id in recent_training_requests:
                last_request = recent_training_requests[admin_id]
                time_diff = now - last_request
                
                if time_diff < 5:
                    print(f"⏭️ Duplicate training request from {admin_id} (within {time_diff:.2f}s)")
                    return jsonify({
                        'success': False,
                        'error': 'Training request submitted too quickly'
                    }), 429
            
            recent_training_requests[admin_id] = now
        
        # Check for existing active jobs
        from supabase import create_client
        supabase = create_client(config.SUPABASE['url'], config.SUPABASE['key'])
        
        active_jobs = supabase.table('training_jobs')\
            .select('job_id, status')\
            .eq('admin_id', admin_id)\
            .in_('status', ['training'])\
            .execute()
        
        if active_jobs.data and len(active_jobs.data) > 0:
            print(f"❌ Admin {admin_id} already has active job")
            return jsonify({
                'success': False,
                'error': 'You already have a training in progress'
            }), 400
        
        # Auto-cancel any active countdown
        countdown_manager.cancel_countdown(admin_id)
        
        print(f"\n🚀 Manual Training: {len(student_ids)} students (Admin: {admin_id})")
        
        # ✅ CREATE JOB WITH STATUS='training' IMMEDIATELY (same as countdown)
        try:
            job_response = supabase.table('training_jobs').insert({
                'admin_id': admin_id,
                'total_classes': len(student_ids) + 2,
                'status': 'training'  # ✅ Immediate 'training' status
            }).execute()
            
            if not job_response.data:
                raise Exception('Failed to create training job')
            
            job_id = job_response.data[0]['job_id']
            print(f"✅ Training job created with status='training': {job_id}")
            
        except Exception as e:
            print(f"❌ Failed to create training job: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
        
        # ✅ Start training in background (reuse same function)
        thread = threading.Thread(
            target=start_training_background,
            args=(admin_id, student_ids, job_id),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training started',
            'job_id': job_id,
            'total_classes': len(student_ids) + 2,
            'student_classes': len(student_ids),
            'admin_id': admin_id
        })
        
    except Exception as e:
        print(f"❌ Training endpoint error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ===========================
# SUPABASE MODEL ENDPOINTS
# ===========================
@app.route('/models/supabase/latest', methods=['GET'])
def get_latest_supabase_model():
    """Get latest model from Supabase"""
    try:
        if not config.SUPABASE['enabled']:
            return jsonify({'success': False, 'error': 'Supabase not enabled'}), 400
        
        recorder = init_supabase()
        model = recorder.get_latest_model()
        
        if not model:
            return jsonify({'success': False, 'error': 'No models found'}), 404
        
        return jsonify({
            'success': True,
            'model': model
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/models/supabase/<model_uuid>', methods=['GET'])
def get_supabase_model_by_uuid(model_uuid):
    """Get specific model from Supabase by UUID"""
    try:
        if not config.SUPABASE['enabled']:
            return jsonify({'success': False, 'error': 'Supabase not enabled'}), 400
        
        recorder = init_supabase()
        model = recorder.get_model_by_uuid(model_uuid)
        
        if not model:
            return jsonify({'success': False, 'error': f'Model {model_uuid} not found'}), 404
        
        return jsonify({
            'success': True,
            'model': model
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/models/supabase/list', methods=['GET'])
def list_supabase_models():
    """List models from Supabase"""
    try:
        if not config.SUPABASE['enabled']:
            return jsonify({'success': False, 'error': 'Supabase not enabled'}), 400
        
        limit = request.args.get('limit', 10, type=int)
        
        recorder = init_supabase()
        models = recorder.list_models(limit=limit)
        
        return jsonify({
            'success': True,
            'models': models,
            'count': len(models)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
# ===========================
# MODEL MANAGEMENT ENDPOINTS
# ===========================
@app.route('/models/list', methods=['GET'])
def list_models():
    """List all trained models"""
    try:
        models = list_trained_models()
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/download_model', methods=['GET'])
def download_model():
    """Download trained model as ZIP"""
    try:
        import zipfile
        
        # Get latest model path
        latest_model = get_latest_model()
        if not latest_model:
            return jsonify({'error': 'No trained model found'}), 404
        
        # Create ZIP file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add model file
            zip_file.write(latest_model, os.path.basename(latest_model))
            
            # Add metadata file
            metadata_path = latest_model.replace('model_', 'metadata_').replace('.keras', '.json')
            if os.path.exists(metadata_path):
                zip_file.write(metadata_path, os.path.basename(metadata_path))
        
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='signature_model.zip'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# ===========================
# RUN SERVER
# ===========================
if __name__ == '__main__':
    # Create directories
    os.makedirs(config.STORAGE['models_dir'], exist_ok=True)


    
    print("\n" + "="*60)
    print("🚀 SIGNATURE RECOGNITION API SERVER (ALL-IN-ONE PYTHON)")
    print("="*60)
    print(f"📦 Model: {config.MODEL['backbone']}")
    print(f"🔧 Configuration: WebGL-matched")
    print(f"🌐 Server: http://{config.API['host']}:{config.API['port']}")
    print(f"☁️  S3 Operations: Integrated")
    print("="*60)
    print("\n📋 Available Endpoints:")
    print("   Training:")
    print("     POST /train_production - Train from Supabase")
    print("     GET  /training_status - Get training progress")
    print("\n   S3 Operations:")
    print("     POST /api/upload-signature - Upload signature to S3")
    print("     POST /api/generate-signature-url - Generate presigned URL")
    print("     POST /api/generate-signature-urls-batch - Batch presigned URLs")
    print("     DELETE /api/delete-signature - Delete signature from S3")
    print("     POST /api/upload-model-to-s3 - Upload model to S3")
    print("     GET  /api/download-model/<uuid> - Download model from S3")
    print("\n   Model Management:")
    print("     GET  /models/list - List local models")
    print("     GET  /download_model - Download model as ZIP")
    print("     GET  /models/supabase/latest - Get latest from Supabase")
    print("     GET  /models/supabase/<uuid> - Get specific from Supabase")
    print("     GET  /models/supabase/list - List Supabase models")
    print("\n   Utilities:")
    print("     GET  /health - Health check")
    print("     GET  /config - Get configuration")
    print("="*60 + "\n")
    
    # ✅ FIXED: Consistent indentation (4 spaces from if block)
    app.run(
        host=config.API['host'],
        port=config.API['port'],
        debug=config.API['debug']
    )
