#!/usr/bin/env python3
"""
Script para probar la API de recomendación de empleos
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_job_recommendation_api():
    """Probar todos los endpoints de la API de recomendación"""
    print("🧪 Testing Job Recommendation API")
    print(f"Base URL: {BASE_URL}")
    
    # Test 1: Health check
    print("\n1️⃣ Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Status: {health_data['status']}")
            print(f"   Model loaded: {health_data['model_loaded']}")
            print(f"   Total jobs: {health_data['total_jobs']}")
        else:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 2: Search methods info
    print("\n2️⃣ Testing search methods...")
    try:
        response = requests.get(f"{BASE_URL}/search-methods")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            methods = response.json()
            print("   ✅ Available methods:")
            for method, info in methods['methods'].items():
                print(f"      - {method}: {info['name']}")
            print(f"   Recommended method: {methods['recommended']}")
        else:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Get all jobs (pagination)
    print("\n3️⃣ Testing get all jobs...")
    try:
        response = requests.get(f"{BASE_URL}/jobs?skip=0&limit=5")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            jobs_data = response.json()
            print(f"   ✅ Found {jobs_data['total']} total jobs")
            print(f"   Showing {len(jobs_data['jobs'])} jobs:")
            for job in jobs_data['jobs']:
                print(f"      - {job['title']} at {job['company']}")
        else:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Job recommendations con diferentes métodos
    print("\n4️⃣ Testing job recommendations...")
    test_queries = [
        {"query": "Data Scientist", "top_n": 5, "method": "hybrid"},
        {"query": "Software Engineer", "top_n": 3, "method": "title_only"},
        {"query": "Product Manager", "top_n": 4, "method": "combined"}
    ]
    
    for test_query in test_queries:
        print(f"\n   🔍 Query: '{test_query['query']}' (method: {test_query['method']})")
        try:
            response = requests.post(f"{BASE_URL}/recommend", json=test_query)
            print(f"      Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"      ✅ Success: {result['success']}")
                print(f"      Found {result['total_results']} recommendations")
                print(f"      Method used: {result['method']}")
                
                for rec in result['recommendations'][:3]:  # Solo mostrar top 3
                    print(f"      {rec['rank']}. {rec['title']} at {rec['company']} (Score: {rec['similarity_score']:.3f})")
                    
            else:
                print(f"      Response: {response.json()}")
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    # Test 5: Get specific job details
    print("\n5️⃣ Testing specific job details...")
    try:
        # Primero obtener un job ID de la lista
        response = requests.get(f"{BASE_URL}/jobs?skip=0&limit=1")
        if response.status_code == 200:
            first_job = response.json()['jobs'][0]
            job_id = first_job['id']
            
            response = requests.get(f"{BASE_URL}/jobs/{job_id}")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                job_details = response.json()
                print(f"   ✅ Job details for ID {job_id}:")
                print(f"      Title: {job_details['title']}")
                print(f"      Company: {job_details['company']}")
                print(f"      Location: {job_details['location']}")
                print(f"      Description: {job_details['description'][:100]}...")
            else:
                print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 6: Error handling - job not found
    print("\n6️⃣ Testing error handling...")
    try:
        response = requests.get(f"{BASE_URL}/jobs/999999")
        print(f"   Status: {response.status_code}")
        if response.status_code == 404:
            print("   ✅ Correctly handled non-existent job")
        else:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 7: Performance test
    print("\n7️⃣ Testing performance...")
    try:
        start_time = time.time()
        test_query = {"query": "Machine Learning Engineer", "top_n": 10, "method": "hybrid"}
        response = requests.post(f"{BASE_URL}/recommend", json=test_query)
        end_time = time.time()
        
        print(f"   Status: {response.status_code}")
        print(f"   ⏱️ Response time: {(end_time - start_time):.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Found {result['total_results']} recommendations")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n🎉 API testing completed!")

if __name__ == "__main__":
    test_job_recommendation_api()
