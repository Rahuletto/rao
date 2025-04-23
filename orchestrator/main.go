package rao

import (
	"fmt"
	"io/ioutil"
	"log"
	gemini "rao/utils"
	"strings"
	"sync"
	"time"
)

type Agent struct {
	Type    string `json:"type"`
	Usecase string `json:"usecase"`
	System  string `json:"system"`
	Prompt  string `json:"prompt"`
	Model   string `json:"model"`
}

type AgentResult struct {
	Type       string        `json:"type"`
	Query      string        `json:"query"`
	Response   string        `json:"response"`
	Error      error         `json:"-"`
	ErrorMsg   string        `json:"error,omitempty"`
	Model      string        `json:"model"`
	Duration   time.Duration `json:"duration"`
	StartTime  time.Time     `json:"start_time"`
	FinishTime time.Time     `json:"finish_time"`
}

type ProcessSummary struct {
	TotalAgents     int           `json:"total_agents"`
	SuccessfulRuns  int           `json:"successful_runs"`
	FailedRuns      int           `json:"failed_runs"`
	TotalDuration   time.Duration `json:"total_duration"`
	StartTime       time.Time     `json:"start_time"`
	FinishTime      time.Time     `json:"finish_time"`
	MasterAgentType string        `json:"master_agent_type"`
	FinalAgentType  string        `json:"final_agent_type"`
}

type OrchestrationResult struct {
	MasterPrompt  string                 `json:"master_prompt"`
	Process       *ProcessSummary        `json:"process"`
	AgentResults  []*AgentResult         `json:"agent_results"`
	FinalResponse map[string]interface{} `json:"final_response"`
}

type StatusUpdate struct {
	Type     string      `json:"type"`
	Message  string      `json:"message"`
	Data     interface{} `json:"data,omitempty"`
	Metadata interface{} `json:"metadata,omitempty"`
	Time     time.Time   `json:"timestamp"`
}

type Orchestrator struct {
	client *gemini.GeminiClient
}

func NewOrchestrator() (*Orchestrator, error) {
	client, err := gemini.NewGeminiClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %v", err)
	}
	return &Orchestrator{client: client}, nil
}

func (o *Orchestrator) RunAgentsStreaming(masterPrompt string, systemPrompt string, masterModelType string, updateChan chan<- *StatusUpdate) (*OrchestrationResult, error) {
	processStart := time.Now()
	result := &OrchestrationResult{
		MasterPrompt: masterPrompt,
		Process: &ProcessSummary{
			StartTime: processStart,
		},
	}

	updateChan <- &StatusUpdate{
		Type:    "master_started",
		Message: "Master agent processing started",
		Data: map[string]interface{}{
			"prompt": masterPrompt,
			"model":  masterModelType,
		},
		Time: time.Now(),
	}

	masterResponse, err := o.client.GenerateObject(masterModelType, masterPrompt, systemPrompt, true)
	if err != nil {
		updateChan <- &StatusUpdate{
			Type:    "error",
			Message: fmt.Sprintf("Failed to generate master response: %v", err),
			Time:    time.Now(),
		}
		return nil, fmt.Errorf("failed to generate master response: %v", err)
	}

	updateChan <- &StatusUpdate{
		Type:    "master_completed",
		Message: "Master agent processing completed",
		Data:    masterResponse,
		Time:    time.Now(),
	}

	if masterType, ok := masterResponse["type"].(string); ok {
		result.Process.MasterAgentType = masterType
	} else {
		result.Process.MasterAgentType = "Unknown Master Agent"
	}

	agentsData, ok := masterResponse["agents"].([]interface{})
	if !ok {
		updateChan <- &StatusUpdate{
			Type:    "error",
			Message: "Master response doesn't contain valid agents array",
			Time:    time.Now(),
		}
		return nil, fmt.Errorf("master response doesn't contain valid agents array")
	}

	var agents []Agent
	for _, agentData := range agentsData {
		agentMap, ok := agentData.(map[string]interface{})
		if !ok {
			updateChan <- &StatusUpdate{
				Type:    "error",
				Message: "Invalid agent format",
				Time:    time.Now(),
			}
			return nil, fmt.Errorf("invalid agent format")
		}

		agent := Agent{
			Type:    agentMap["type"].(string),
			Usecase: agentMap["usecase"].(string),
			System:  agentMap["system"].(string),
			Prompt:  agentMap["prompt"].(string),
			Model:   agentMap["model"].(string),
		}
		agents = append(agents, agent)
	}

	updateChan <- &StatusUpdate{
		Type:    "agents_created",
		Message: fmt.Sprintf("Created %d agents for processing", len(agents)),
		Data:    agents,
		Time:    time.Now(),
	}

	agentResults := o.runAgentsParallelStreaming(agents, updateChan)
	result.AgentResults = agentResults

	result.Process.TotalAgents = len(agents)
	result.Process.SuccessfulRuns = 0
	result.Process.FailedRuns = 0

	for _, ar := range agentResults {
		if ar.Error != nil {
			result.Process.FailedRuns++
		} else {
			result.Process.SuccessfulRuns++
		}
	}

	updateChan <- &StatusUpdate{
		Type: "agents_completed",
		Message: fmt.Sprintf("All agents completed: %d successful, %d failed",
			result.Process.SuccessfulRuns, result.Process.FailedRuns),
		Metadata: map[string]interface{}{
			"successful": result.Process.SuccessfulRuns,
			"failed":     result.Process.FailedRuns,
			"total":      result.Process.TotalAgents,
		},
		Time: time.Now(),
	}

	updateChan <- &StatusUpdate{
		Type:    "final_processing_started",
		Message: "Starting final unbiased synthesis with LearnLM",
		Time:    time.Now(),
	}

	finalPrompt := o.buildFinalLearnLMPrompt(masterPrompt, agentResults)
	finalSystemPrompt := `You are LearnLM, an unbiased research and synthesis AI. Your task is to analyze all provided agent responses and create a comprehensive, unbiased final output that integrates all perspectives. Do not favor any specific agent or perspective. Present a balanced view that considers all input equally. Focus on factual information and clearly distinguish between consensus views and areas of disagreement. Do not add any personal opinions or biases. Your goal is to provide the most objective and comprehensive synthesis possible.`

	finalResponse, err := o.client.GenerateObject(string(gemini.LearnLM), finalPrompt, finalSystemPrompt, false)

	if err != nil {
		updateChan <- &StatusUpdate{
			Type:    "final_processing_error",
			Message: fmt.Sprintf("Error during final synthesis: %v", err),
			Time:    time.Now(),
		}

		allResponses := make(map[string]interface{})
		allResponses["type"] = "Combined Agent Responses (Fallback)"
		allResponses["query"] = masterPrompt

		agentResponsesMap := make(map[string]string)
		for _, ar := range agentResults {
			if ar.Error == nil {
				agentResponsesMap[ar.Type] = ar.Response
			}
		}

		allResponses["response"] = "All agent responses provided without merging or filtering (fallback due to synthesis error)."
		allResponses["agent_responses"] = agentResponsesMap

		result.FinalResponse = allResponses
		result.Process.FinalAgentType = "Direct Agent Response Collection (Fallback)"
	} else {
		updateChan <- &StatusUpdate{
			Type:    "final_processing_completed",
			Message: "Completed final unbiased synthesis with LearnLM",
			Data:    finalResponse,
			Time:    time.Now(),
		}

		finalResponse["agent_responses_raw"] = o.collectRawAgentResponses(agentResults)
		result.FinalResponse = finalResponse
		result.Process.FinalAgentType = "LearnLM Unbiased Synthesis"
	}

	result.Process.FinishTime = time.Now()
	result.Process.TotalDuration = result.Process.FinishTime.Sub(result.Process.StartTime)

	updateChan <- &StatusUpdate{
		Type:    "process_completed",
		Message: "Orchestration process completed successfully",
		Data: map[string]interface{}{
			"duration": result.Process.TotalDuration.String(),
		},
		Time: time.Now(),
	}

	return result, nil
}

func (o *Orchestrator) RunAgents(masterPrompt string, systemPrompt string, masterModelType string) (*OrchestrationResult, error) {
	processStart := time.Now()
	result := &OrchestrationResult{
		MasterPrompt: masterPrompt,
		Process: &ProcessSummary{
			StartTime: processStart,
		},
	}

	masterResponse, err := o.client.GenerateObject(masterModelType, masterPrompt, systemPrompt, true)
	if err != nil {
		return nil, fmt.Errorf("failed to generate master response: %v", err)
	}

	if masterType, ok := masterResponse["type"].(string); ok {
		result.Process.MasterAgentType = masterType
	} else {
		result.Process.MasterAgentType = "Unknown Master Agent"
	}

	agentsData, ok := masterResponse["agents"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("master response doesn't contain valid agents array")
	}

	var agents []Agent
	for _, agentData := range agentsData {
		agentMap, ok := agentData.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid agent format")
		}

		agent := Agent{
			Type:    agentMap["type"].(string),
			Usecase: agentMap["usecase"].(string),
			System:  agentMap["system"].(string),
			Prompt:  agentMap["prompt"].(string),
			Model:   agentMap["model"].(string),
		}
		agents = append(agents, agent)
	}

	agentResults := o.runAgentsParallel(agents)
	result.AgentResults = agentResults

	result.Process.TotalAgents = len(agents)
	result.Process.SuccessfulRuns = 0
	result.Process.FailedRuns = 0

	for _, ar := range agentResults {
		if ar.Error != nil {
			result.Process.FailedRuns++
		} else {
			result.Process.SuccessfulRuns++
		}
	}

	finalPrompt := o.buildFinalLearnLMPrompt(masterPrompt, agentResults)
	finalSystemPrompt := `You are LearnLM, an unbiased research and synthesis AI. Your task is to analyze all provided agent responses and create a comprehensive, unbiased final output that integrates all perspectives. Do not favor any specific agent or perspective. Present a balanced view that considers all input equally. Focus on factual information and clearly distinguish between consensus views and areas of disagreement. Do not add any personal opinions or biases. Your goal is to provide the most objective and comprehensive synthesis possible.`

	finalResponse, err := o.client.GenerateObject(string(gemini.LearnLM), finalPrompt, finalSystemPrompt, false)

	if err != nil {

		allResponses := make(map[string]interface{})
		allResponses["type"] = "Combined Agent Responses (Fallback)"
		allResponses["query"] = masterPrompt

		agentResponsesMap := make(map[string]string)
		for _, ar := range agentResults {
			if ar.Error == nil {
				agentResponsesMap[ar.Type] = ar.Response
			}
		}

		allResponses["response"] = "All agent responses provided without merging or filtering (fallback due to synthesis error)."
		allResponses["agent_responses"] = agentResponsesMap

		result.FinalResponse = allResponses
		result.Process.FinalAgentType = "Direct Agent Response Collection (Fallback)"
	} else {
		finalResponse["agent_responses_raw"] = o.collectRawAgentResponses(agentResults)
		result.FinalResponse = finalResponse
		result.Process.FinalAgentType = "LearnLM Unbiased Synthesis"
	}

	result.Process.FinishTime = time.Now()
	result.Process.TotalDuration = result.Process.FinishTime.Sub(result.Process.StartTime)

	return result, nil
}

func (o *Orchestrator) collectRawAgentResponses(agentResults []*AgentResult) map[string]string {
	agentResponsesMap := make(map[string]string)
	for _, ar := range agentResults {
		if ar.Error == nil {
			agentResponsesMap[ar.Type] = ar.Response
		}
	}
	return agentResponsesMap
}

func (o *Orchestrator) runAgentsParallel(agents []Agent) []*AgentResult {
	var wg sync.WaitGroup
	resultChan := make(chan *AgentResult, len(agents))

	for _, agent := range agents {
		wg.Add(1)

		agentCopy := agent

		go func() {
			defer wg.Done()

			result := &AgentResult{
				Type:      agentCopy.Type,
				Query:     agentCopy.Prompt,
				Model:     agentCopy.Model,
				StartTime: time.Now(),
			}

			systemPrompt := agentCopy.System
			if strings.Contains(strings.ToLower(agentCopy.Type), "cod") ||
				strings.Contains(strings.ToLower(agentCopy.Usecase), "cod") {

				content, err := ioutil.ReadFile("prompts/codex.txt")
				if err != nil {
					log.Printf("Error loading codex prompt: %v", err)
				} else {
					systemPrompt = string(content)
				}
			}

			resp, err := o.client.GenerateObject(agentCopy.Model, agentCopy.Prompt, systemPrompt, false)

			result.FinishTime = time.Now()
			result.Duration = result.FinishTime.Sub(result.StartTime)

			if err != nil {
				result.Error = err
				result.ErrorMsg = err.Error()
				resultChan <- result
				return
			}

			if response, ok := resp["response"].(string); ok {
				result.Response = response
			} else {
				result.Error = fmt.Errorf("invalid response format from agent")
				result.ErrorMsg = "invalid response format from agent"
			}

			resultChan <- result
		}()
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	var results []*AgentResult
	for result := range resultChan {
		results = append(results, result)
	}

	return results
}

func (o *Orchestrator) runAgentsParallelStreaming(agents []Agent, updateChan chan<- *StatusUpdate) []*AgentResult {
	var wg sync.WaitGroup
	resultChan := make(chan *AgentResult, len(agents))

	for _, agent := range agents {
		wg.Add(1)

		agentCopy := agent

		go func() {
			defer wg.Done()

			result := &AgentResult{
				Type:      agentCopy.Type,
				Query:     agentCopy.Prompt,
				Model:     agentCopy.Model,
				StartTime: time.Now(),
			}

			updateChan <- &StatusUpdate{
				Type:    "agent_started",
				Message: fmt.Sprintf("Agent %s started processing", agentCopy.Type),
				Data: map[string]interface{}{
					"agent_type": agentCopy.Type,
					"model":      agentCopy.Model,
				},
				Time: time.Now(),
			}

			systemPrompt := agentCopy.System
			if strings.Contains(strings.ToLower(agentCopy.Type), "cod") ||
				strings.Contains(strings.ToLower(agentCopy.Usecase), "cod") {

				content, err := ioutil.ReadFile("prompts/codex.txt")
				if err != nil {
					log.Printf("Error loading codex prompt: %v", err)
				} else {
					systemPrompt = string(content)
					updateChan <- &StatusUpdate{
						Type:    "agent_prompt_change",
						Message: fmt.Sprintf("Using codex.txt for coding agent %s", agentCopy.Type),
						Time:    time.Now(),
					}
				}
			}

			resp, err := o.client.GenerateObject(agentCopy.Model, agentCopy.Prompt, systemPrompt, false)

			result.FinishTime = time.Now()
			result.Duration = result.FinishTime.Sub(result.StartTime)

			if err != nil {
				result.Error = err
				result.ErrorMsg = err.Error()

				updateChan <- &StatusUpdate{
					Type:    "agent_error",
					Message: fmt.Sprintf("Agent %s encountered an error: %v", agentCopy.Type, err),
					Data: map[string]interface{}{
						"agent_type": agentCopy.Type,
						"error":      err.Error(),
					},
					Time: time.Now(),
				}

				resultChan <- result
				return
			}

			if response, ok := resp["response"].(string); ok {
				result.Response = response
			} else {
				result.Error = fmt.Errorf("invalid response format from agent")
				result.ErrorMsg = "invalid response format from agent"

				updateChan <- &StatusUpdate{
					Type:    "agent_error",
					Message: fmt.Sprintf("Agent %s returned invalid response format", agentCopy.Type),
					Data: map[string]interface{}{
						"agent_type": agentCopy.Type,
					},
					Time: time.Now(),
				}

				resultChan <- result
				return
			}

			updateChan <- &StatusUpdate{
				Type:    "agent_completed",
				Message: fmt.Sprintf("Agent %s completed successfully in %v", agentCopy.Type, result.Duration),
				Data: map[string]interface{}{
					"agent_type": agentCopy.Type,
					"duration":   result.Duration.String(),
				},
				Time: time.Now(),
			}

			resultChan <- result
		}()
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	var results []*AgentResult
	for result := range resultChan {
		results = append(results, result)
	}

	return results
}

func (o *Orchestrator) buildFinalLearnLMPrompt(originalPrompt string, results []*AgentResult) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("ORIGINAL QUERY: %s\n\n", originalPrompt))
	sb.WriteString("AGENT RESPONSES:\n\n")

	for _, result := range results {
		if result.Error != nil {
			continue
		}

		sb.WriteString(fmt.Sprintf("AGENT: %s\n", result.Type))
		sb.WriteString(fmt.Sprintf("MODEL: %s\n", result.Model))
		sb.WriteString(fmt.Sprintf("RESPONSE:\n%s\n\n", result.Response))
	}

	sb.WriteString(`
	TASK: Analyze all agent responses provided above and produce a final, fully synthesized, actionable output that directly answers the original query with research evidences. Integrate all relevant information and perspectives from the agents equally—do not favor any single response. 

Your response should not reflect on summary or the inputs—instead, deliver a clear, structured, and technically accurate final result as if you were the final decision-maker. Combine the best ideas, resolve overlaps or conflicts, and generate a unified, high-value deliverable for the user. 

This is not a commentary—this is the final product.

`)

	return sb.String()
}
