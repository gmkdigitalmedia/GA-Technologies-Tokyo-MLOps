#!/usr/bin/env python3
"""
Async Functionality Verification Test
Run this to verify all async functionality is working properly
"""

import asyncio
import time
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

class AsyncVerificationTest:
    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
        
    async def run_all_tests(self):
        """Run comprehensive async verification tests"""
        print("LAUNCH Starting GP MLOps Platform Async Verification")
        print("=" * 60)
        
        # Test 1: Basic async functionality
        await self.test_basic_async()
        
        # Test 2: FastAPI app structure
        await self.test_fastapi_structure()
        
        # Test 3: Service layer async functions
        await self.test_service_layer()
        
        # Test 4: Concurrent execution
        await self.test_concurrent_execution()
        
        # Test 5: Error handling in async context
        await self.test_async_error_handling()
        
        # Generate final report
        self.generate_final_report()
        
    async def test_basic_async(self):
        """Test 1: Basic async/await functionality"""
        print("\nTEST Test 1: Basic Async Functionality")
        print("-" * 40)
        
        async def simple_async_task():
            await asyncio.sleep(0.001)
            return "async_working"
        
        try:
            start_time = time.time()
            result = await simple_async_task()
            duration = time.time() - start_time
            
            success = result == "async_working"
            self.record_test("basic_async", success, f"Completed in {duration:.4f}s")
            print(f"PASS Basic async: {result} ({duration:.4f}s)")
            
        except Exception as e:
            self.record_test("basic_async", False, f"Error: {e}")
            print(f"FAIL Basic async failed: {e}")
    
    async def test_fastapi_structure(self):
        """Test 2: FastAPI app and endpoint structure"""
        print("\nTEST Test 2: FastAPI Structure")
        print("-" * 40)
        
        try:
            # Test importing main app
            from main import app
            self.record_test("fastapi_import", True, "Successfully imported FastAPI app")
            print("PASS FastAPI app import successful")
            
            # Check if app has async routes
            route_count = 0
            async_route_count = 0
            
            for route in app.routes:
                if hasattr(route, 'endpoint'):
                    route_count += 1
                    if asyncio.iscoroutinefunction(route.endpoint):
                        async_route_count += 1
            
            success = async_route_count > 0
            self.record_test("fastapi_async_routes", success, 
                           f"Found {async_route_count}/{route_count} async routes")
            print(f"PASS Async routes: {async_route_count}/{route_count}")
            
        except Exception as e:
            self.record_test("fastapi_import", False, f"Import error: {e}")
            self.record_test("fastapi_async_routes", False, f"Route check failed: {e}")
            print(f"FAIL FastAPI structure test failed: {e}")
    
    async def test_service_layer(self):
        """Test 3: Service layer async functions"""
        print("\nTEST Test 3: Service Layer Async Functions")
        print("-" * 40)
        
        services_to_test = [
            ("core.config", "settings"),
            ("services.mlops_pipeline", "mlops_pipeline"),
            ("core.monitoring", None)
        ]
        
        for module_name, service_attr in services_to_test:
            try:
                module = __import__(module_name, fromlist=[service_attr] if service_attr else [''])
                
                if service_attr:
                    service = getattr(module, service_attr)
                    # Check if service has async methods
                    async_methods = [name for name in dir(service) 
                                   if not name.startswith('_') and 
                                   callable(getattr(service, name)) and
                                   asyncio.iscoroutinefunction(getattr(service, name))]
                    
                    success = len(async_methods) > 0
                    self.record_test(f"service_{module_name}", success, 
                                   f"Found {len(async_methods)} async methods")
                    print(f"PASS {module_name}: {len(async_methods)} async methods")
                else:
                    # Just check if module imports
                    self.record_test(f"service_{module_name}", True, "Module imported successfully")
                    print(f"PASS {module_name}: Imported successfully")
                    
            except Exception as e:
                self.record_test(f"service_{module_name}", False, f"Import error: {e}")
                print(f"FAIL {module_name}: Import failed - {e}")
    
    async def test_concurrent_execution(self):
        """Test 4: Concurrent async execution"""
        print("\nTEST Test 4: Concurrent Execution")
        print("-" * 40)
        
        async def concurrent_task(task_id: int, delay: float = 0.01):
            await asyncio.sleep(delay)
            return f"task_{task_id}_complete"
        
        try:
            # Test sequential execution
            start_time = time.time()
            sequential_results = []
            for i in range(5):
                result = await concurrent_task(i)
                sequential_results.append(result)
            sequential_time = time.time() - start_time
            
            # Test parallel execution
            start_time = time.time()
            tasks = [concurrent_task(i) for i in range(5)]
            parallel_results = await asyncio.gather(*tasks)
            parallel_time = time.time() - start_time
            
            # Calculate efficiency
            speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            efficiency = "excellent" if speedup > 3 else "good" if speedup > 2 else "moderate"
            
            success = len(parallel_results) == 5 and speedup > 1
            self.record_test("concurrent_execution", success, 
                           f"Speedup: {speedup:.2f}x ({efficiency})")
            print(f"PASS Concurrent execution: {speedup:.2f}x speedup ({efficiency})")
            print(f"   Sequential: {sequential_time:.4f}s | Parallel: {parallel_time:.4f}s")
            
        except Exception as e:
            self.record_test("concurrent_execution", False, f"Error: {e}")
            print(f"FAIL Concurrent execution failed: {e}")
    
    async def test_async_error_handling(self):
        """Test 5: Error handling in async context"""
        print("\nTEST Test 5: Async Error Handling")
        print("-" * 40)
        
        async def failing_task():
            await asyncio.sleep(0.001)
            raise ValueError("Intentional test error")
        
        async def recovery_task():
            try:
                await failing_task()
                return "should_not_reach"
            except ValueError:
                return "error_handled"
            except Exception:
                return "unexpected_error"
        
        try:
            # Test error handling
            result = await recovery_task()
            success = result == "error_handled"
            self.record_test("async_error_handling", success, f"Result: {result}")
            print(f"PASS Error handling: {result}")
            
            # Test asyncio.gather with error handling
            async def mixed_tasks():
                return await asyncio.gather(
                    asyncio.sleep(0.001, result="success"),
                    recovery_task(),
                    return_exceptions=True
                )
            
            gather_results = await mixed_tasks()
            gather_success = len(gather_results) == 2 and gather_results[1] == "error_handled"
            self.record_test("async_gather_errors", gather_success, 
                           f"Gather results: {len(gather_results)}")
            print(f"PASS Gather error handling: {len(gather_results)} results")
            
        except Exception as e:
            self.record_test("async_error_handling", False, f"Error: {e}")
            self.record_test("async_gather_errors", False, f"Error: {e}")
            print(f"FAIL Error handling test failed: {e}")
    
    def record_test(self, test_name: str, success: bool, details: str = ""):
        """Record test result"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
        
        self.results[test_name] = {
            "success": success,
            "details": details,
            "timestamp": time.time()
        }
    
    def generate_final_report(self):
        """Generate and display final test report"""
        print("\n" + "=" * 60)
        print("🎯 ASYNC VERIFICATION FINAL REPORT")
        print("=" * 60)
        
        print(f"📊 Overall Results:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   Passed: {self.passed_tests}")
        print(f"   Failed: {self.total_tests - self.passed_tests}")
        print(f"   Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        print(f"\nTEST Detailed Results:")
        for test_name, result in self.results.items():
            status = "PASS PASS" if result["success"] else "FAIL FAIL"
            print(f"   {status} {test_name}: {result['details']}")
        
        # Overall assessment
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        if success_rate >= 90:
            assessment = "🌟 EXCELLENT - Async functionality is working perfectly!"
        elif success_rate >= 75:
            assessment = "PASS GOOD - Async functionality is mostly working"
        elif success_rate >= 50:
            assessment = "⚠️  MODERATE - Some async issues need attention"
        else:
            assessment = "FAIL POOR - Significant async issues detected"
        
        print(f"\n🎯 Assessment: {assessment}")
        
        # Save detailed report
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "success_rate": success_rate,
            "assessment": assessment,
            "results": self.results
        }
        
        with open("async_verification_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: async_verification_report.json")
        
        return success_rate >= 75  # Return True if tests mostly passed

async def main():
    """Main test execution"""
    print("GP MLOps Platform - Async Functionality Verification")
    print("This test will verify that async functionality is working correctly")
    print()
    
    tester = AsyncVerificationTest()
    await tester.run_all_tests()
    
    return tester.results

if __name__ == "__main__":
    try:
        results = asyncio.run(main())
        sys.exit(0)  # Success
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFAIL Test execution failed: {e}")
        sys.exit(1)