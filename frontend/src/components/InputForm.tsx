import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { SquarePen, Brain, Send, StopCircle, Cpu, RefreshCw } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// Updated InputFormProps
interface InputFormProps {
  onSubmit: (inputValue: string, effort: string, model: string) => void;
  onCancel: () => void;
  isLoading: boolean;
  hasHistory: boolean;
}

const MODEL_LIST_STORAGE_KEY = "research-agent:model-list";
const SELECTED_MODEL_STORAGE_KEY = "research-agent:selected-model";

function getCachedModelList(fallbackModels: string[]): string[] {
  if (typeof window === "undefined") {
    return fallbackModels;
  }

  const cachedModels = window.localStorage.getItem(MODEL_LIST_STORAGE_KEY);
  if (!cachedModels) {
    return fallbackModels;
  }

  try {
    const parsedModels = JSON.parse(cachedModels) as string[];
    return parsedModels.length > 0 ? parsedModels : fallbackModels;
  } catch {
    return fallbackModels;
  }
}

function getInitialSelectedModel(availableModels: string[], fallbackModel: string): string {
  if (typeof window === "undefined") {
    return fallbackModel;
  }

  const cachedSelectedModel = window.localStorage.getItem(
    SELECTED_MODEL_STORAGE_KEY
  );
  if (cachedSelectedModel && availableModels.includes(cachedSelectedModel)) {
    return cachedSelectedModel;
  }

  return availableModels[0] ?? fallbackModel;
}

export const InputForm: React.FC<InputFormProps> = ({
  onSubmit,
  onCancel,
  isLoading,
  hasHistory,
}) => {
  const fallbackModels = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
  ];
  const initialModels = getCachedModelList(fallbackModels);
  const [internalInputValue, setInternalInputValue] = useState("");
  const [effort, setEffort] = useState("medium");
  const [availableModels, setAvailableModels] = useState<string[]>(initialModels);
  const [model, setModel] = useState(
    getInitialSelectedModel(initialModels, "gemini-2.5-flash")
  );
  const [isRefreshingModels, setIsRefreshingModels] = useState(false);
  const [hasInitializedModelState, setHasInitializedModelState] = useState(false);

  useEffect(() => {
    const cachedModels = window.localStorage.getItem(MODEL_LIST_STORAGE_KEY);
    const cachedSelectedModel = window.localStorage.getItem(
      SELECTED_MODEL_STORAGE_KEY
    );

    if (cachedModels && availableModels.length > 0) {
      setHasInitializedModelState(true);
      return;
    }

    if (cachedSelectedModel) {
      setModel(cachedSelectedModel);
    }
    setHasInitializedModelState(true);

    void refreshModels();
  }, []);

  useEffect(() => {
    if (!hasInitializedModelState) {
      return;
    }
    window.localStorage.setItem(SELECTED_MODEL_STORAGE_KEY, model);
  }, [hasInitializedModelState, model]);

  const refreshModels = async () => {
    try {
      setIsRefreshingModels(true);

      const response = await fetch("/api/models", {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      const data = (await response.json()) as {
        models?: string[];
        default_model?: string | null;
      };
      const models = data.models?.filter(Boolean) ?? [];
      if (!models.length) {
        return;
      }

      setAvailableModels(models);
      window.localStorage.setItem(MODEL_LIST_STORAGE_KEY, JSON.stringify(models));

      setModel((currentModel) => {
        const cachedSelectedModel = window.localStorage.getItem(
          SELECTED_MODEL_STORAGE_KEY
        );
        const resolvedModel =
          currentModel && models.includes(currentModel)
            ? currentModel
            : cachedSelectedModel && models.includes(cachedSelectedModel)
              ? cachedSelectedModel
              : data.default_model && models.includes(data.default_model)
                ? data.default_model
                : models[0];
        window.localStorage.setItem(SELECTED_MODEL_STORAGE_KEY, resolvedModel);
        return resolvedModel;
      });
    } catch {
    } finally {
      setIsRefreshingModels(false);
    }
  };

  const handleModelChange = (nextModel: string) => {
    if (!nextModel) {
      return;
    }
    setModel(nextModel);
  };

  const handleInternalSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!internalInputValue.trim()) return;
    onSubmit(internalInputValue, effort, model);
    setInternalInputValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit with Ctrl+Enter (Windows/Linux) or Cmd+Enter (Mac)
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleInternalSubmit();
    }
  };

  const isSubmitDisabled = !internalInputValue.trim() || isLoading;

  return (
    <form
      onSubmit={handleInternalSubmit}
      className={`flex flex-col gap-2 p-3 pb-4`}
    >
      <div
        className={`flex flex-row items-center justify-between text-white rounded-3xl rounded-bl-sm ${
          hasHistory ? "rounded-br-sm" : ""
        } break-words min-h-7 bg-neutral-700 px-4 pt-3 `}
      >
        <Textarea
          value={internalInputValue}
          onChange={(e) => setInternalInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Who won the Euro 2024 and scored the most goals?"
          className={`w-full text-neutral-100 placeholder-neutral-500 resize-none border-0 focus:outline-none focus:ring-0 outline-none focus-visible:ring-0 shadow-none
                        md:text-base  min-h-[56px] max-h-[200px]`}
          rows={1}
        />
        <div className="-mt-3">
          {isLoading ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="text-red-500 hover:text-red-400 hover:bg-red-500/10 p-2 cursor-pointer rounded-full transition-all duration-200"
              onClick={onCancel}
            >
              <StopCircle className="h-5 w-5" />
            </Button>
          ) : (
            <Button
              type="submit"
              variant="ghost"
              className={`${
                isSubmitDisabled
                  ? "text-neutral-500"
                  : "text-blue-500 hover:text-blue-400 hover:bg-blue-500/10"
              } p-2 cursor-pointer rounded-full transition-all duration-200 text-base`}
              disabled={isSubmitDisabled}
            >
              Search
              <Send className="h-5 w-5" />
            </Button>
          )}
        </div>
      </div>
      <div className="flex items-center justify-between">
        <div className="flex flex-row gap-2">
          <div className="flex flex-row gap-2 bg-neutral-700 border-neutral-600 text-neutral-300 focus:ring-neutral-500 rounded-xl rounded-t-sm pl-2  max-w-[100%] sm:max-w-[90%]">
            <div className="flex flex-row items-center text-sm">
              <Brain className="h-4 w-4 mr-2" />
              Effort
            </div>
            <Select value={effort} onValueChange={setEffort}>
              <SelectTrigger className="w-[120px] bg-transparent border-none cursor-pointer">
                <SelectValue placeholder="Effort" />
              </SelectTrigger>
              <SelectContent className="bg-neutral-700 border-neutral-600 text-neutral-300 cursor-pointer">
                <SelectItem
                  value="low"
                  className="hover:bg-neutral-600 focus:bg-neutral-600 cursor-pointer"
                >
                  Low
                </SelectItem>
                <SelectItem
                  value="medium"
                  className="hover:bg-neutral-600 focus:bg-neutral-600 cursor-pointer"
                >
                  Medium
                </SelectItem>
                <SelectItem
                  value="high"
                  className="hover:bg-neutral-600 focus:bg-neutral-600 cursor-pointer"
                >
                  High
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-row gap-2 bg-neutral-700 border-neutral-600 text-neutral-300 focus:ring-neutral-500 rounded-xl rounded-t-sm pl-2  max-w-[100%] sm:max-w-[90%]">
            <div className="flex flex-row items-center text-sm ml-2">
              <Cpu className="h-4 w-4 mr-2" />
              Model
            </div>
            <Select value={model} onValueChange={handleModelChange}>
              <SelectTrigger className="w-[150px] bg-transparent border-none cursor-pointer">
                <SelectValue placeholder="Model" />
              </SelectTrigger>
              <SelectContent className="bg-neutral-700 border-neutral-600 text-neutral-300 cursor-pointer">
                {availableModels.map((availableModel) => (
                  <SelectItem
                    key={availableModel}
                    value={availableModel}
                    className="hover:bg-neutral-600 focus:bg-neutral-600 cursor-pointer"
                  >
                    <div className="flex items-center">
                      <Cpu className="h-4 w-4 mr-2 text-sky-400" />
                      {availableModel}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="cursor-pointer text-neutral-300 hover:text-white hover:bg-neutral-600/60"
              onClick={() => void refreshModels()}
              disabled={isRefreshingModels}
              title="Reload models"
            >
              <RefreshCw
                className={`h-4 w-4 ${isRefreshingModels ? "animate-spin" : ""}`}
              />
            </Button>
          </div>
        </div>
        {hasHistory && (
          <Button
            className="bg-neutral-700 border-neutral-600 text-neutral-300 cursor-pointer rounded-xl rounded-t-sm pl-2 "
            variant="default"
            onClick={() => window.location.reload()}
          >
            <SquarePen size={16} />
            New Search
          </Button>
        )}
      </div>
    </form>
  );
};
