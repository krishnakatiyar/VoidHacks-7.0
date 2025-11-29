import json
import os
from alzheimer_agent import AlzheimerAgent

# Test Cases
TEST_CASES = [
    # LOW RISK (4 cases)
    {"id": "low_1", "inputs": ["No memory issues", "No", "No", "No", "No"]},
    {"id": "low_2", "inputs": ["I sleep well", "No family history", "No confusion", "No daily task issues", "No"]},
    {"id": "low_3", "inputs": ["Just tired", "No forgetting", "No", "No", "No"]},
    {"id": "low_4", "inputs": ["Healthy", "No", "No", "No", "No"]},
    
    # MODERATE RISK (3 cases)
    {"id": "mod_1", "inputs": ["I forget names sometimes", "6 months", "No daily issues", "My dad had it", "No"]},
    {"id": "mod_2", "inputs": ["Trouble with bills", "Yes sometimes", "No confusion", "No family history", "No"]},
    {"id": "mod_3", "inputs": ["Lost keys often", "Yes", "No", "No", "Yes diabetes"]},
    
    # HIGH RISK (3 cases)
    {"id": "high_1", "inputs": ["Forget conversations daily", "2 years", "Major trouble with tasks", "Yes confused often", "Mom had alzheimers"]},
    {"id": "high_2", "inputs": ["Lost driving home", "Yes often", "Yes daily tasks hard", "Yes", "Yes"]},
    {"id": "high_3", "inputs": ["Don't know where I am", "Yes", "Yes", "Yes", "Yes"]}
]

def run_test(case):
    print(f"\n--- Running Case {case['id']} ---")
    agent = AlzheimerAgent(session_id=f"test_{case['id']}")
    
    # Initial input
    resp = agent.step(case['inputs'][0])
    print(f"Turn 1: {resp['type']}")
    
    # Follow-ups
    for i, inp in enumerate(case['inputs'][1:]):
        if resp['type'] == 'final_report':
            break
        resp = agent.step(inp)
        print(f"Turn {i+2}: {resp['type']}")
        
    if resp['type'] == 'final_report':
        print(f"Result: {resp['risk_level']} (Prob: {resp['probability']})")
        return resp
    else:
        print("Result: Incomplete (Max rounds reached)")
        return None

if __name__ == "__main__":
    results = []
    for case in TEST_CASES:
        res = run_test(case)
        if res:
            results.append({"id": case['id'], "risk": res['risk_level'], "prob": res['probability']})
            
    print("\n--- Summary ---")
    print(json.dumps(results, indent=2))
