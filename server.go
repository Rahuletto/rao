package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	rao "rao/orchestrator"
	gemini "rao/utils"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/websocket/v2"
)

var system string

type SessionManager struct {
	sessions map[string]*Session
	mu       sync.RWMutex
}

type Session struct {
	ID       string
	UpdateCh chan *rao.StatusUpdate
	Done     chan struct{}
}

var sessionManager = SessionManager{
	sessions: make(map[string]*Session),
}

func (sm *SessionManager) CreateSession(id string) *Session {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	session := &Session{
		ID:       id,
		UpdateCh: make(chan *rao.StatusUpdate, 100),
		Done:     make(chan struct{}),
	}

	sm.sessions[id] = session
	return session
}

func (sm *SessionManager) GetSession(id string) (*Session, bool) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	session, exists := sm.sessions[id]
	return session, exists
}

func (sm *SessionManager) CloseSession(id string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if session, exists := sm.sessions[id]; exists {
		close(session.UpdateCh)
		close(session.Done)
		delete(sm.sessions, id)
	}
}

func init() {
	content, err := ioutil.ReadFile("prompts/system.txt")
	if err != nil {
		log.Fatalf("Error loading system prompt: %v", err)
	}
	system = string(content)
}

func main() {
	app := fiber.New()

	app.Use(cors.New(cors.Config{
		AllowOrigins: "*",
		AllowHeaders: "Origin, Content-Type, Accept, Authorization",
		AllowMethods: "GET, POST, PUT, DELETE, OPTIONS",
	}))

	app.Use("/ws", func(c *fiber.Ctx) error {

		if websocket.IsWebSocketUpgrade(c) {
			c.Locals("allowed", true)
			return c.Next()
		}
		return fiber.ErrUpgradeRequired
	})

	app.Get("/", func(c *fiber.Ctx) error {
		return c.SendString("Hello, World!")
	})

	app.Post("/api/gemini", func(c *fiber.Ctx) error {
		var requestBody map[string]interface{}
		if err := c.BodyParser(&requestBody); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Invalid request body",
			})
		}
		client, err := gemini.NewGeminiClient()
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": err,
			})
		}
		prompt, ok := requestBody["prompt"].(string)
		if !ok {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Invalid prompt format",
			})
		}
		resp, err := client.GenerateObject(string(gemini.Gemini_2_5_Pro), prompt, system, true)
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": err,
			})
		}
		return c.JSON(resp)
	})

	app.Post("/api/agents", func(c *fiber.Ctx) error {
		var requestBody map[string]interface{}
		if err := c.BodyParser(&requestBody); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Invalid request body",
			})
		}

		prompt, ok := requestBody["prompt"].(string)
		if !ok {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Invalid prompt format",
			})
		}

		modelType := string(gemini.Gemini_2_5_Pro)
		if model, ok := requestBody["model"].(string); ok && model != "" {
			modelType = model
		}

		orchestrator, err := rao.NewOrchestrator()
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": "Failed to create orchestrator: " + err.Error(),
			})
		}

		orchestrationResult, err := orchestrator.RunAgents(prompt, system, modelType)
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": "Failed to run agents: " + err.Error(),
			})
		}

		return c.JSON(orchestrationResult)
	})

	app.Get("/ws/agents/:sessionID", websocket.New(func(c *websocket.Conn) {

		sessionID := c.Params("sessionID")
		log.Printf("WebSocket connection established for session: %s", sessionID)

		session, exists := sessionManager.GetSession(sessionID)
		if !exists {

			c.WriteJSON(fiber.Map{
				"type":    "error",
				"message": "Invalid session ID or session expired",
			})
			return
		}

		go func() {
			defer sessionManager.CloseSession(sessionID)

			for {
				var msg map[string]interface{}
				if err := c.ReadJSON(&msg); err != nil {

					log.Printf("WebSocket connection closed for session %s: %v", sessionID, err)
					return
				}

				log.Printf("Received message from client: %v", msg)
			}
		}()

		for {
			select {
			case update, open := <-session.UpdateCh:
				if !open {

					return
				}
				if err := c.WriteJSON(update); err != nil {
					log.Printf("Error sending update: %v", err)
					return
				}
			case <-session.Done:

				return
			}
		}
	}))

	app.Post("/api/agents/stream", func(c *fiber.Ctx) error {
		var requestBody map[string]interface{}
		if err := c.BodyParser(&requestBody); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Invalid request body",
			})
		}

		prompt, ok := requestBody["prompt"].(string)
		if !ok {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Invalid prompt format",
			})
		}

		modelType := string(gemini.Gemini_2_5_Pro)
		if model, ok := requestBody["model"].(string); ok && model != "" {
			modelType = model
		}

		sessionID := generateSessionID()
		session := sessionManager.CreateSession(sessionID)

		go func() {
			defer sessionManager.CloseSession(sessionID)

			orchestrator, err := rao.NewOrchestrator()
			if err != nil {
				session.UpdateCh <- &rao.StatusUpdate{
					Type:    "error",
					Message: "Failed to create orchestrator: " + err.Error(),
					Time:    time.Now(),
				}
				return
			}

			result, err := orchestrator.RunAgentsStreaming(prompt, system, modelType, session.UpdateCh)
			if err != nil {

				log.Printf("Error running streaming orchestration: %v", err)
			} else {

				finalResultBytes, _ := json.Marshal(result)
				finalResultStr := string(finalResultBytes)

				if len(finalResultStr) > 1000 {
					finalResultStr = finalResultStr[:1000] + "... (truncated)"
				}

				session.UpdateCh <- &rao.StatusUpdate{
					Type:    "final_result",
					Message: "Final orchestration result",
					Data:    result,
					Time:    time.Now(),
				}

				close(session.Done)
			}
		}()

		return c.JSON(fiber.Map{
			"session_id": sessionID,
			"message":    "Orchestration started. Connect to WebSocket to receive updates.",
		})
	})

	app.Listen(":3000")
}

func generateSessionID() string {

	return fmt.Sprintf("sess_%d", time.Now().UnixNano())
}
