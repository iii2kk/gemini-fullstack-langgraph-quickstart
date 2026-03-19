import { useState, useCallback, useRef } from "react";

export interface Message {
  type: "human" | "ai";
  content: string;
  id: string;
}

interface SubmitValues {
  messages: Message[];
  initial_search_query_count: number;
  max_research_loops: number;
  reasoning_model: string;
}

interface UseResearchOptions {
  onUpdateEvent?: (event: Record<string, unknown>) => void;
  onError?: (error: Error) => void;
}

interface UseResearchResult {
  messages: Message[];
  isLoading: boolean;
  submit: (values: SubmitValues) => void;
  stop: () => void;
}

export function useResearch(options: UseResearchOptions = {}): UseResearchResult {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const submit = useCallback(
    (values: SubmitValues) => {
      // Show human messages immediately
      setMessages(values.messages);
      setIsLoading(true);

      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      (async () => {
        try {
          const response = await fetch("/api/research/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(values),
            signal: abortController.signal,
          });

          if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
          }

          const reader = response.body!.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() ?? "";

            for (const line of lines) {
              if (!line.startsWith("data: ")) continue;
              const data = line.slice(6).trim();
              if (data === "[DONE]") return;

              let event: Record<string, unknown>;
              try {
                event = JSON.parse(data);
              } catch {
                continue;
              }

              if (event.error) {
                options.onError?.(new Error(String(event.error)));
                return;
              }

              if (event.messages) {
                // Final AI messages
                const newMsgs = event.messages as Message[];
                setMessages((prev) => [...prev, ...newMsgs]);
              } else {
                // Intermediate progress events
                options.onUpdateEvent?.(event);
              }
            }
          }
        } catch (err) {
          if (err instanceof Error && err.name === "AbortError") return;
          options.onError?.(err instanceof Error ? err : new Error(String(err)));
        } finally {
          setIsLoading(false);
        }
      })();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const stop = useCallback(() => {
    abortControllerRef.current?.abort();
    setIsLoading(false);
  }, []);

  return { messages, isLoading, submit, stop };
}
