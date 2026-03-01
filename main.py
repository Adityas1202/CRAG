import os
import sys
from src.graph import build_graph

# This ensures your project root is in the python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def chat():
    # Build the compiled LangGraph application
    app = build_graph()
    
    print("\n" + "="*29)
    print("      🚀 CRAG SYSTEM 🚀      ")
    print("="*29)
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        user_input = input("\nUser: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Shutting down CRAG system. Goodbye!")
            break
            
        # We use app.stream to see the process step-by-step
        inputs = {"question": user_input}
        
        for output in app.stream(inputs):
            for node_name, state_update in output.items():
                # Visual separator for nodes
                #print(f"\n[ENTRY: {node_name.upper()}]")
                
                # Special Debug: See what the web search actually found
                if node_name == "web_search":
                    content = state_update.get("documents", "")
                    #print(f"--- DEBUG: WEB CONTEXT ACQUIRED ---")
                    # Show first 300 characters of the search result
                   # print(f"{content[:300]}...") 
                
                # When we hit the final generation node, print the AI response
                if node_name == "generate":
                    print("\n" + "-"*10 + " AI RESPONSE " + "-"*10)
                    print(state_update.get("generation"))
                    print("-"*33)

if __name__ == "__main__":
    try:
        chat()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")