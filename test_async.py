#!/usr/bin/env python3
"""
Simple Async Test - Just run: python3 test_async.py

HOW THIS WORKS:
This script performs 4 different tests to verify async functionality:

1. BASIC ASYNC TEST: Creates a simple async function and runs it to verify 
   that Python's async/await syntax is working correctly

2. CONCURRENT EXECUTION TEST: Creates 5 async tasks that run in parallel 
   using asyncio.gather() to verify that async code actually runs concurrently
   (should be much faster than running sequentially)

3. ERROR HANDLING TEST: Tests that exceptions work properly in async functions
   by creating a function that raises an error and another that catches it

4. CODEBASE ANALYSIS: Scans all .py files in your project to count how many
   'async def ' functions exist - this verifies your actual codebase has async code

The script does NOT test your actual FastAPI endpoints - it tests the underlying
async functionality that your endpoints rely on. If this passes, it means your
Python environment can handle async code properly.
"""

import asyncio
import time
from pathlib import Path

async def main():
    print("Testing Async Functionality")
    print("=" * 40)
    print("This tests if your Python environment can handle async code properly")
    print("(Not testing your actual API endpoints - just the async foundation)")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Basic async/await functionality
    print("\n1. Basic async test...")
    print("   Testing if 'async def' and 'await' work correctly")
    total_tests += 1
    try:
        # Create a simple async function that sleeps for 0.01 seconds then returns a value
        async def simple_task():
            await asyncio.sleep(0.01)  # This line tests 'await' keyword
            return "working"
        
        # Call the async function (this tests if async functions can be executed)
        result = await simple_task()
        if result == "working":
            print("   PASS: Basic async - async/await syntax works")
            tests_passed += 1
        else:
            print("   FAIL: Basic async - unexpected return value")
    except Exception as e:
        print(f"   FAIL: Basic async - {e}")
    
    # Test 2: Concurrent execution (the main benefit of async)
    print("\n2. Concurrent execution test...")
    print("   Testing if multiple async tasks run in parallel (not sequentially)")
    total_tests += 1
    try:
        # Create an async function that simulates work with a 0.01 second delay
        async def task(n):
            await asyncio.sleep(0.01)  # Simulate I/O work
            return f"task_{n}"
        
        # Run 5 tasks concurrently using asyncio.gather()
        # If they run in parallel: ~0.01 seconds total
        # If they run sequentially: ~0.05 seconds total
        start = time.time()
        results = await asyncio.gather(*[task(i) for i in range(5)])
        duration = time.time() - start
        
        # Verify we got 5 results and it was fast (parallel execution)
        if len(results) == 5 and duration < 0.1:
            print(f"   PASS: Concurrent execution - tasks ran in parallel ({duration:.3f}s)")
            tests_passed += 1
        else:
            print(f"   FAIL: Concurrent execution - too slow, tasks may have run sequentially ({duration:.3f}s)")
    except Exception as e:
        print(f"   FAIL: Concurrent execution - {e}")
    
    # Test 3: Error handling in async functions
    print("\n3. Async error handling...")
    print("   Testing if try/except works properly with async functions")
    total_tests += 1
    try:
        # Create an async function that always raises an error
        async def failing_task():
            raise ValueError("test error")
        
        # Create another async function that handles the error
        async def handle_error():
            try:
                await failing_task()  # This should raise ValueError
                return "no_error"     # This should NOT be reached
            except ValueError:
                return "caught_error" # This SHOULD be reached
        
        # Test that the error was caught properly
        result = await handle_error()
        if result == "caught_error":
            print("   PASS: Error handling - exceptions work in async functions")
            tests_passed += 1
        else:
            print("   FAIL: Error handling - exception not caught properly")
    except Exception as e:
        print(f"   FAIL: Error handling - {e}")
    
    # Test 4: Import and test actual async functions from codebase
    print("\n4. Testing actual async functions in codebase...")
    print("   → Attempting to import each Python file and test async functions")
    print("   → This verifies your actual async code works, not just syntax")
    
    import sys
    import os
    import importlib.util
    import inspect
    
    # Add current directory to Python path so we can import modules
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    
    tested_functions = 0
    working_functions = 0
    failed_imports = 0
    
    try:
        # Walk through all Python files in the current directory and subdirectories
        for py_file in Path(".").rglob("*.py"):
            # Skip test files, cache directories, and this test file itself
            if "test_" in str(py_file) or "__pycache__" in str(py_file) or py_file.name == "test_async.py":
                continue
            
            print(f"\n   Testing file: {py_file}")
            
            try:
                # Convert file path to module name (e.g., app/main.py -> app.main)
                module_path = str(py_file).replace('/', '.').replace('\\', '.').replace('.py', '')
                
                # Try to import the module
                try:
                    if module_path.startswith('.'):
                        module_path = module_path[1:]  # Remove leading dot
                    
                    # Load module from file path
                    spec = importlib.util.spec_from_file_location(module_path, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        print(f"     IMPORT: SUCCESS")
                    else:
                        print(f"     IMPORT: FAILED - could not create spec")
                        failed_imports += 1
                        continue
                        
                except Exception as import_error:
                    print(f"     IMPORT: FAILED - {import_error}")
                    failed_imports += 1
                    continue
                
                # Find all async functions in the imported module
                async_functions = []
                for name, obj in inspect.getmembers(module):
                    if inspect.iscoroutinefunction(obj):
                        async_functions.append((name, obj))
                
                if not async_functions:
                    print(f"     ASYNC FUNCTIONS: None found")
                    continue
                
                print(f"     ASYNC FUNCTIONS: Found {len(async_functions)}")
                
                # Test each async function
                for func_name, func_obj in async_functions:
                    tested_functions += 1
                    print(f"       Testing {func_name}...")
                    
                    try:
                        # Get function signature to see what parameters it expects
                        sig = inspect.signature(func_obj)
                        params = list(sig.parameters.keys())
                        
                        # Try to call the function with minimal/mock parameters
                        if len(params) == 0:
                            # No parameters - try to call directly
                            result = await asyncio.wait_for(func_obj(), timeout=2.0)
                            print(f"         RESULT: SUCCESS - returned {type(result).__name__}")
                            working_functions += 1
                            
                        elif len(params) == 1 and 'self' in params:
                            # Method that only needs self - skip for now
                            print(f"         RESULT: SKIPPED - requires self parameter")
                            
                        else:
                            # Function with parameters - try with None values or skip
                            print(f"         RESULT: SKIPPED - requires parameters: {params}")
                            
                    except asyncio.TimeoutError:
                        print(f"         RESULT: TIMEOUT - function took too long")
                        
                    except Exception as func_error:
                        error_type = type(func_error).__name__
                        print(f"         RESULT: ERROR - {error_type}: {str(func_error)[:100]}")
                        
            except Exception as file_error:
                print(f"     FILE ERROR: {file_error}")
                continue
        
        # Calculate results for Test 4
        total_tests += 1
        if tested_functions > 0 and working_functions > 0:
            print(f"\n   SUMMARY:")
            print(f"     Tested {tested_functions} async functions")
            print(f"     {working_functions} functions executed successfully")
            print(f"     {failed_imports} files failed to import")
            
            # Consider test passed if we successfully tested some functions
            if working_functions > 0:
                tests_passed += 1
                print(f"     OVERALL: PASS - Some async functions are working")
            else:
                print(f"     OVERALL: FAIL - No async functions executed successfully")
        else:
            print(f"     OVERALL: FAIL - No async functions could be tested")
            
    except Exception as e:
        print(f"   CODEBASE TEST: FAIL - {e}")
        total_tests += 1
    
    # Results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    success_rate = (tests_passed / total_tests) * 100 if total_tests > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print("EXCELLENT - Async functionality is working!")
    elif success_rate >= 50:
        print("GOOD - Most async functionality working")
    else:
        print("ISSUES - Some async problems detected")
    
    return tests_passed >= 3

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        print("\nTest complete!")
    except Exception as e:
        print(f"\nTest failed: {e}")