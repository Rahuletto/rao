import asyncio
import json
import re
from dotenv import load_dotenv

import pydantic
from typedef.main import MasterAgentConfig
from utils.agent import CreateMaster, FinalVerdict, CreateChild
from agno.agent import RunResponse
from agno.utils.pprint import pprint_run_response


async def run_child_agent(agent_config, index):
    print(f"[{index}] Creating and running agent: {agent_config.type}")
    
    child_agent = CreateChild(
        model=agent_config.model,
        system=agent_config.system,
    )
    
    child_result = await child_agent.arun(agent_config.prompt)
    
    print(f"\n--- Result from {agent_config.type} ---")
    pprint_run_response(child_result)
    
    return {
        "config": agent_config,
        "result": child_result.content,
        "index": index
    }

query = input("Enter your query: ")

async def main():
    load_dotenv()
    
    readFile = open("./prompts/system.txt", "r")
    system = readFile.read()
    if system == "":
        raise ValueError("System file is empty. Please provide a valid system file.")
    
    master_agent = CreateMaster("gemini-2.0-flash", system, MasterAgentConfig)
    
    try:
        print("Running master agent...")
        master_result: RunResponse = master_agent.run(query)
        pprint_run_response(master_result)
        
        if isinstance(master_result.content, str) and "```json" in master_result.content:
            print("Detected JSON in markdown format. Processing...")
            
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', master_result.content)
            if json_match:
                json_str = json_match.group(1).strip()
                try:
                    json_data = json.loads(json_str)
                    master_config = pydantic.TypeAdapter.validate_json(MasterAgentConfig, json_data)
                    print("Successfully parsed JSON markdown into MasterAgentConfig")
                except json.JSONDecodeError as je:
                    print(f"Error parsing JSON: {je}")
                    raise
                except Exception as e:
                    print(f"Error converting to MasterAgentConfig: {e}")
                    raise
            else:
                print("Could not extract JSON from markdown")
                raise ValueError("Invalid response format")
        else:
            master_config: MasterAgentConfig = master_result.content
        
        print(f"Spawning {len(master_config.agents)} child agents in parallel...")
        tasks = [
            run_child_agent(agent_config, i) 
            for i, agent_config in enumerate(master_config.agents)
        ]
        
        child_results = await asyncio.gather(*tasks)
        
        child_results.sort(key=lambda x: x["index"])
        
        verdict_system = """
        You are LearnLM, an unbiased research and synthesis AI. Your task is to analyze all provided agent responses and create a comprehensive, unbiased final output that integrates all perspectives. Do not favor any specific agent or perspective. Present a balanced view that considers all input equally. Focus on factual information and clearly distinguish between consensus views and areas of disagreement. Do not add any personal opinions or biases. Your goal is to provide the most objective and comprehensive synthesis possible.
        """
        
        verdict_prompt = f"""
        ORIGINAL QUERY: {query}
        """
        
        for result in child_results:
            config = result["config"]
            content = result["result"]
            
            
            verdict_prompt += f"""
                AGENT: {config.type}
                FOCUS: {config.usecase}
                
                OUTPUT:
                {content}
                """
            
        verdict_prompt += """
        TASK: Analyze all agent responses provided above and produce a final, fully synthesized, actionable output that directly answers the original query with research evidences. Integrate all relevant information and perspectives from the agents equally—do not favor any single response. 
        Your response should not reflect on summary or the inputs—instead, deliver a clear, structured, and technically accurate final result as if you were the final decision-maker. Combine the best ideas, resolve overlaps or conflicts, and generate a unified, high-value deliverable for the user. 
        This is not a commentary—this is the final product.
        """
        
        print("\nCreating final verdict agent...")
        verdict_agent = FinalVerdict(
            model="learnlm-2.0-flash-experimental",
            system=verdict_system,
        )
        
        verdict_agent.print_response(verdict_prompt, stream=True, markdown=True)
        
        
    except Exception as e:
        print(f"Error running agent pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())