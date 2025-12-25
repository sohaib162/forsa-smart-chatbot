import { useTypingEffect } from "@/hooks/use-typing-effect";
import { MarkdownMessage } from "./MarkdownMessage";

interface TypingMarkdownMessageProps {
  content: string;
  isTyping?: boolean;
}

export function TypingMarkdownMessage({ content, isTyping }: TypingMarkdownMessageProps) {
  const { displayedText, isTyping: currentlyTyping } = useTypingEffect({
    text: content,
    speed: 15,
    enabled: isTyping || false
  });

  return (
    <div className="relative">
      <MarkdownMessage content={displayedText} />
      {currentlyTyping && (
        <span className="inline-block w-0.5 h-4 bg-primary ml-0.5 animate-pulse align-middle" />
      )}
    </div>
  );
}
