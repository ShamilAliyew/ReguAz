import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiService } from "../services/api";
import { useChatStore } from "../stores/useChatStore";

export function useChat() {
  const queryClient = useQueryClient();
  const { 
    activeSessionId, 
    addUserMessage, 
    appendStreamChunk, 
    finalizeMessage,
    setIsGenerating,
    selectedModel,
  } = useChatStore();

  const sendMessageMutation = useMutation({
    mutationFn: async (question: string) => {
      setIsGenerating(true);
      
      // 1. Add user message to local state
      addUserMessage(question);
      
      // Create a unique temporary message ID for the assistant streaming response
      const assistantMessageId = "msg-assistant-" + Date.now();
      
      // Initialize assistant placeholder message in store
      appendStreamChunk(assistantMessageId, ""); 

      try {
        // 2. Query backend / mock service
        const response = await apiService.postChat(
          question,
          activeSessionId,
          selectedModel,
        );
        
        // 3. Simulate streaming the response tokens to the UI
        const text = response.answer;
        const words = text.split(" ");
        let currentIndex = 0;
        
        // We'll write chunks incrementally
        return new Promise<typeof response>((resolve) => {
          const interval = setInterval(() => {
            if (currentIndex < words.length) {
              const chunk = words[currentIndex] + (currentIndex < words.length - 1 ? " " : "");
              appendStreamChunk(assistantMessageId, chunk);
              currentIndex++;
            } else {
              clearInterval(interval);
              // Finalize message with correct sources & metrics
              finalizeMessage(
                assistantMessageId, 
                response.answer, 
                response.sources, 
                response.metrics,
                response.citations,
                response.pipeline_version,
                response.warnings,
                response.model,
              );
              setIsGenerating(false);
              resolve(response);
            }
          }, 35); // 35ms per word
        });

      } catch (error) {
        setIsGenerating(false);
        finalizeMessage(
          assistantMessageId,
          "Xəta baş verdi: Normativ aktların araşdırılması zamanı serverlə əlaqə qurulmadı. Zəhmət olmasa tənzimləmə parametrlərini yoxlayın.",
          []
        );
        throw error;
      }
    },
    onSuccess: () => {
      // Invalidate history to refresh lists
      queryClient.invalidateQueries({ queryKey: ["chat-history"] });
    }
  });

  return {
    sendMessage: sendMessageMutation.mutate,
    isPending: sendMessageMutation.isPending,
    error: sendMessageMutation.error,
  };
}
