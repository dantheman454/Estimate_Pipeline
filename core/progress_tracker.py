"""
Progress Tracking System for Blueprint Pipeline Upgrade
Provides real-time progress updates and status monitoring during implementation
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class TaskStatus(Enum):
    """Task completion status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Individual task tracking data structure."""
    id: str
    name: str
    description: str
    phase: str
    estimated_time_minutes: int
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Calculate actual task duration in minutes."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() / 60
        return None
    
    @property
    def is_ready(self) -> bool:
        """Check if task dependencies are satisfied."""
        # For now, assume all dependencies are satisfied
        # This could be enhanced to check actual dependency completion
        return True


class ProgressTracker:
    """
    Comprehensive progress tracking system for blueprint pipeline upgrade.
    
    Features:
    - Phase-based progress tracking
    - Individual task monitoring
    - Time estimation and actual duration tracking
    - JSON-based persistence
    - Console output formatting
    """
    
    def __init__(self, project_name: str = "Blueprint Pipeline Upgrade"):
        self.project_name = project_name
        self.tasks: Dict[str, Task] = {}
        self.phases: List[str] = []
        self.start_time = datetime.now()
        self.callbacks: List[Callable] = []
        
        # Progress file for persistence
        self.progress_file = Path("upgrade_progress.json")
        
        # Initialize with upgrade plan tasks
        self._initialize_upgrade_tasks()
        
        # Load existing progress if available
        self._load_progress()
    
    def _initialize_upgrade_tasks(self):
        """Initialize all tasks from the upgrade plan."""
        
        # Phase 1: Core Infrastructure Setup
        phase1_tasks = [
            Task("p1_update_deps", "Update Dependencies", 
                 "Add OpenCV, Pytesseract, and other required libraries", 
                 "Phase 1", 30),
            Task("p1_create_image_processor", "Create Image Processing Module", 
                 "Implement legend detection and region extraction", 
                 "Phase 1", 90),
            Task("p1_create_symbol_extractor", "Create Symbol Extraction Module", 
                 "Implement symbol detection and classification", 
                 "Phase 1", 60),
        ]
        
        # Phase 2: Enhanced AI Detection
        phase2_tasks = [
            Task("p2_upgrade_smolvlm", "Upgrade SmolVLM Detector", 
                 "Implement dual-image processing with contextual prompts", 
                 "Phase 2", 120),
            Task("p2_remove_obsolete", "Remove Obsolete Components", 
                 "Clean up PDF-specific processing code", 
                 "Phase 2", 60),
        ]
        
        # Phase 3: CLI Interface Upgrade
        phase3_tasks = [
            Task("p3_create_image_cli", "Create New Image Processing CLI", 
                 "Build process_image.py with progress tracking", 
                 "Phase 3", 90),
            Task("p3_update_main_cli", "Update Main CLI", 
                 "Add backward compatibility and format detection", 
                 "Phase 3", 30),
        ]
        
        # Phase 4: Progress Tracking System
        phase4_tasks = [
            Task("p4_implement_progress", "Implement Progress Tracking", 
                 "Create progress tracking module (this task)", 
                 "Phase 4", 45),
            Task("p4_integrate_progress", "Integrate Progress Tracking", 
                 "Add progress tracking to all major operations", 
                 "Phase 4", 15),
        ]
        
        # Phase 5: Testing and Validation
        phase5_tasks = [
            Task("p5_create_tests", "Create Test Suite", 
                 "Build comprehensive test suite for new functionality", 
                 "Phase 5", 90),
            Task("p5_validate_images", "Validation with Provided Images", 
                 "Test with key.png and Full_plan.png", 
                 "Phase 5", 60),
        ]
        
        # Phase 6: Documentation and Cleanup
        phase6_tasks = [
            Task("p6_update_docs", "Update Documentation", 
                 "Update README and create usage examples", 
                 "Phase 6", 45),
            Task("p6_cleanup", "Clean Up Obsolete Code", 
                 "Remove unused functions and imports", 
                 "Phase 6", 15),
        ]
        
        # Combine all tasks
        all_tasks = (phase1_tasks + phase2_tasks + phase3_tasks + 
                    phase4_tasks + phase5_tasks + phase6_tasks)
        
        # Add to tasks dictionary
        for task in all_tasks:
            self.tasks[task.id] = task
        
        # Set up phases
        self.phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5", "Phase 6"]
    
    def start_task(self, task_id: str) -> bool:
        """Start a specific task."""
        if task_id not in self.tasks:
            print(f"❌ Task not found: {task_id}")
            return False
        
        task = self.tasks[task_id]
        
        if not task.is_ready:
            print(f"⏳ Task dependencies not satisfied: {task.name}")
            return False
        
        task.status = TaskStatus.IN_PROGRESS
        task.start_time = datetime.now()
        
        print(f"🚀 Started: {task.name}")
        self._save_progress()
        self._notify_callbacks("task_started", task)
        
        return True
    
    def complete_task(self, task_id: str, success: bool = True, error_message: str = None) -> bool:
        """Complete a specific task."""
        if task_id not in self.tasks:
            print(f"❌ Task not found: {task_id}")
            return False
        
        task = self.tasks[task_id]
        task.end_time = datetime.now()
        
        if success:
            task.status = TaskStatus.COMPLETED
            print(f"✅ Completed: {task.name}")
            if task.duration_minutes:
                print(f"   Duration: {task.duration_minutes:.1f} minutes")
        else:
            task.status = TaskStatus.FAILED
            task.error_message = error_message
            print(f"❌ Failed: {task.name}")
            if error_message:
                print(f"   Error: {error_message}")
        
        self._save_progress()
        self._notify_callbacks("task_completed", task)
        
        return True
    
    def skip_task(self, task_id: str, reason: str = None) -> bool:
        """Skip a specific task."""
        if task_id not in self.tasks:
            print(f"❌ Task not found: {task_id}")
            return False
        
        task = self.tasks[task_id]
        task.status = TaskStatus.SKIPPED
        task.error_message = reason
        
        print(f"⏭️  Skipped: {task.name}")
        if reason:
            print(f"   Reason: {reason}")
        
        self._save_progress()
        self._notify_callbacks("task_skipped", task)
        
        return True
    
    def get_phase_progress(self, phase: str) -> Dict:
        """Get progress statistics for a specific phase."""
        phase_tasks = [task for task in self.tasks.values() if task.phase == phase]
        
        if not phase_tasks:
            return {"total": 0, "completed": 0, "in_progress": 0, "failed": 0, "percentage": 0}
        
        total = len(phase_tasks)
        completed = len([t for t in phase_tasks if t.status == TaskStatus.COMPLETED])
        in_progress = len([t for t in phase_tasks if t.status == TaskStatus.IN_PROGRESS])
        failed = len([t for t in phase_tasks if t.status == TaskStatus.FAILED])
        
        percentage = (completed / total) * 100 if total > 0 else 0
        
        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "failed": failed,
            "percentage": percentage,
            "tasks": phase_tasks
        }
    
    def get_overall_progress(self) -> Dict:
        """Get overall project progress statistics."""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        in_progress_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS])
        failed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
        
        overall_percentage = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        # Calculate time estimates
        total_estimated_minutes = sum(task.estimated_time_minutes for task in self.tasks.values())
        completed_estimated_minutes = sum(
            task.estimated_time_minutes for task in self.tasks.values() 
            if task.status == TaskStatus.COMPLETED
        )
        
        # Calculate actual time spent
        actual_minutes = sum(
            task.duration_minutes or 0 for task in self.tasks.values() 
            if task.duration_minutes is not None
        )
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "failed_tasks": failed_tasks,
            "overall_percentage": overall_percentage,
            "total_estimated_minutes": total_estimated_minutes,
            "completed_estimated_minutes": completed_estimated_minutes,
            "actual_minutes": actual_minutes,
            "project_start_time": self.start_time,
            "current_time": datetime.now()
        }
    
    def print_status_report(self):
        """Print a comprehensive status report."""
        print(f"\n{'='*60}")
        print(f"📊 {self.project_name} - Progress Report")
        print(f"{'='*60}")
        
        overall = self.get_overall_progress()
        
        print(f"Overall Progress: {overall['overall_percentage']:.1f}%")
        print(f"Tasks: {overall['completed_tasks']}/{overall['total_tasks']} completed")
        
        if overall['actual_minutes'] > 0:
            print(f"Time Spent: {overall['actual_minutes']:.1f} minutes")
        
        print(f"Estimated Total Time: {overall['total_estimated_minutes']} minutes")
        
        print(f"\n📋 Phase Breakdown:")
        for phase in self.phases:
            phase_data = self.get_phase_progress(phase)
            status_icon = "✅" if phase_data['percentage'] == 100 else "🔄" if phase_data['in_progress'] > 0 else "⏳"
            print(f"  {status_icon} {phase}: {phase_data['percentage']:.1f}% ({phase_data['completed']}/{phase_data['total']})")
        
        # Show current tasks
        in_progress = [task for task in self.tasks.values() if task.status == TaskStatus.IN_PROGRESS]
        if in_progress:
            print(f"\n🔄 Currently In Progress:")
            for task in in_progress:
                duration = ""
                if task.start_time:
                    elapsed = datetime.now() - task.start_time
                    duration = f" ({elapsed.total_seconds()/60:.1f}min)"
                print(f"  • {task.name}{duration}")
        
        # Show recent completions
        recent_completions = [
            task for task in self.tasks.values() 
            if task.status == TaskStatus.COMPLETED and task.end_time 
            and (datetime.now() - task.end_time).total_seconds() < 3600  # Last hour
        ]
        if recent_completions:
            print(f"\n✅ Recently Completed:")
            for task in recent_completions[-3:]:  # Show last 3
                print(f"  • {task.name}")
        
        print(f"\n{'='*60}")
    
    def add_callback(self, callback: Callable):
        """Add a callback function for progress events."""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, event_type: str, task: Task):
        """Notify all registered callbacks of progress events."""
        for callback in self.callbacks:
            try:
                callback(event_type, task)
            except Exception as e:
                print(f"⚠️  Callback error: {e}")
    
    def _save_progress(self):
        """Save current progress to JSON file."""
        try:
            progress_data = {
                "project_name": self.project_name,
                "start_time": self.start_time.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "tasks": {
                    task_id: {
                        **asdict(task),
                        "start_time": task.start_time.isoformat() if task.start_time else None,
                        "end_time": task.end_time.isoformat() if task.end_time else None,
                        "status": task.status.value
                    }
                    for task_id, task in self.tasks.items()
                }
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Could not save progress: {e}")
    
    def _load_progress(self):
        """Load existing progress from JSON file."""
        if not self.progress_file.exists():
            return
        
        try:
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            
            # Restore task states
            for task_id, task_data in progress_data.get("tasks", {}).items():
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    task.status = TaskStatus(task_data["status"])
                    task.error_message = task_data.get("error_message")
                    
                    if task_data.get("start_time"):
                        task.start_time = datetime.fromisoformat(task_data["start_time"])
                    if task_data.get("end_time"):
                        task.end_time = datetime.fromisoformat(task_data["end_time"])
            
            print(f"📄 Loaded existing progress from {self.progress_file}")
            
        except Exception as e:
            print(f"⚠️  Could not load progress: {e}")


def main():
    """Demo and testing of the progress tracking system."""
    tracker = ProgressTracker()
    
    print("🚀 Blueprint Pipeline Upgrade Progress Tracker Initialized")
    tracker.print_status_report()
    
    # Mark the progress tracker creation as completed
    tracker.start_task("p4_implement_progress")
    time.sleep(1)  # Simulate work
    tracker.complete_task("p4_implement_progress", success=True)
    
    print("\n🎯 Progress Tracking System is now active!")
    print("Use the following methods to track your progress:")
    print("  • tracker.start_task(task_id)")
    print("  • tracker.complete_task(task_id)")
    print("  • tracker.skip_task(task_id, reason)")
    print("  • tracker.print_status_report()")


if __name__ == "__main__":
    main()
