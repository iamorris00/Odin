"""
test_openviking.py
------------------
Verifies that the `tools.py` OpenViking and Gemini integrations
can successfully retrieve L1/L2 summaries and texts.
"""

from src.agents.tools import IADC_SearchTool, VolveHistory_SearchTool

def run_tests():
    print("Initializing Tests...")
    iadc_tool = IADC_SearchTool()
    volve_tool = VolveHistory_SearchTool()
    
    # Test 1: IADC Definition Search
    print("\\n--- Test 1 (IADC Definition) ---")
    res1 = iadc_tool._run("What is non-productive time (NPT)?")
    print("Result snippet:", res1[:500])

    # Test 2: Volve Historical Search
    print("\\n--- Test 2 (Volve Event) ---")
    res2 = volve_tool._run("Did any stuck pipe incidents occur on 15/9-19 A?")
    print("Result snippet:", res2[:500])

if __name__ == "__main__":
    run_tests()
