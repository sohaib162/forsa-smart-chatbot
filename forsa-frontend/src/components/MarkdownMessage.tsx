import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cn } from '@/lib/utils';

interface MarkdownMessageProps {
  content: string;
  className?: string;
}

export function MarkdownMessage({ content, className }: MarkdownMessageProps) {
  return (
    <div
      className={cn(
        "prose prose-sm max-w-none",
        // Base text styling
        "prose-p:text-foreground prose-p:leading-relaxed prose-p:my-2",
        // Headings
        "prose-headings:text-foreground prose-headings:font-semibold prose-headings:mt-6 prose-headings:mb-3",
        "prose-h1:text-2xl prose-h2:text-xl prose-h3:text-lg",
        // Lists
        "prose-ul:my-3 prose-ul:list-disc prose-ul:pl-6",
        "prose-ol:my-3 prose-ol:list-decimal prose-ol:pl-6",
        "prose-li:text-foreground prose-li:my-1 prose-li:leading-relaxed",
        // Strong/Bold
        "prose-strong:text-foreground prose-strong:font-bold",
        // Em/Italic
        "prose-em:text-foreground prose-em:italic",
        // Code
        "prose-code:text-primary prose-code:bg-muted prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:font-mono prose-code:before:content-[''] prose-code:after:content-['']",
        // Code blocks
        "prose-pre:bg-muted prose-pre:border prose-pre:border-border prose-pre:rounded-lg prose-pre:p-4 prose-pre:overflow-x-auto",
        "prose-pre:my-4 prose-pre:text-sm",
        // Links
        "prose-a:text-primary prose-a:underline prose-a:decoration-primary/50 hover:prose-a:decoration-primary prose-a:transition-colors",
        // Blockquotes
        "prose-blockquote:border-l-4 prose-blockquote:border-primary/30 prose-blockquote:pl-4 prose-blockquote:italic prose-blockquote:text-muted-foreground prose-blockquote:my-4",
        // Tables
        "prose-table:border-collapse prose-table:w-full prose-table:my-4",
        "prose-th:border prose-th:border-border prose-th:bg-muted prose-th:px-4 prose-th:py-2 prose-th:text-left prose-th:font-semibold",
        "prose-td:border prose-td:border-border prose-td:px-4 prose-td:py-2",
        // HR
        "prose-hr:border-border prose-hr:my-6",
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Custom renderers for better control
          p: ({ node, ...props }) => <p {...props} />,
          ul: ({ node, ...props }) => <ul {...props} />,
          ol: ({ node, ...props }) => <ol {...props} />,
          li: ({ node, ...props }) => <li {...props} />,
          strong: ({ node, ...props }) => <strong {...props} />,
          em: ({ node, ...props }) => <em {...props} />,
          code: ({ node, inline, ...props }) =>
            inline ? (
              <code {...props} />
            ) : (
              <code className="block" {...props} />
            ),
          a: ({ node, ...props }) => (
            <a {...props} target="_blank" rel="noopener noreferrer" />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
