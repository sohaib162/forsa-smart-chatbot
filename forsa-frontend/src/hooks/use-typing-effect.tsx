import { useState, useEffect, useRef } from "react";

interface UseTypingEffectOptions {
  text: string;
  speed?: number;
  enabled?: boolean;
}

export function useTypingEffect({ text, speed = 25, enabled = true }: UseTypingEffectOptions) {
  const [displayedText, setDisplayedText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const animationRef = useRef<number>();

  useEffect(() => {
    if (!enabled) {
      setDisplayedText(text);
      setIsTyping(false);
      setIsComplete(true);
      return;
    }

    setDisplayedText("");
    setIsTyping(true);
    setIsComplete(false);
    let currentIndex = 0;
    let lastTime = performance.now();

    const animate = (currentTime: number) => {
      const deltaTime = currentTime - lastTime;

      if (deltaTime >= speed) {
        if (currentIndex < text.length) {
          // Add characters in chunks for smoother appearance
          const chunkSize = text[currentIndex] === ' ' ? 1 : Math.random() > 0.7 ? 2 : 1;
          const endIndex = Math.min(currentIndex + chunkSize, text.length);

          setDisplayedText(text.substring(0, endIndex));
          currentIndex = endIndex;
          lastTime = currentTime;
        } else {
          setIsTyping(false);
          setIsComplete(true);
          return;
        }
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [text, speed, enabled]);

  return { displayedText, isTyping, isComplete };
}
