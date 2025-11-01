import { motion } from "framer-motion";
import { FileText, Brain } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface Highlight {
  text: string;
  category: string;
}

interface ServiceExplanation {
  service: string;
  icon: string;
  reasons: string[];
  highlights: string[];
  confidence: number;
  category: string;
}

interface ExplainabilityViewProps {
  originalInput: string;
  highlights: Highlight[];
  explanations: ServiceExplanation[];
}

const categoryColors: Record<string, string> = {
  Compute: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300",
  Storage: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300",
  Database: "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300",
  Networking: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300",
};

export const ExplainabilityView = ({
  originalInput,
  highlights,
  explanations,
}: ExplainabilityViewProps) => {
  const renderHighlightedText = () => {
    let result = originalInput;
    highlights.forEach((highlight) => {
      const colorClass = categoryColors[highlight.category] || "bg-gray-100";
      result = result.replace(
        new RegExp(`(${highlight.text})`, "gi"),
        `<mark class="${colorClass} px-1 py-0.5 rounded">$1</mark>`
      );
    });
    return result;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left: Original Input with Highlights */}
      <motion.div
        className="glass-card rounded-2xl p-6"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <FileText className="text-primary" />
          Your Requirements
        </h3>
        <div
          className="prose prose-sm max-w-none"
          dangerouslySetInnerHTML={{ __html: renderHighlightedText() }}
        />

        <div className="mt-6 p-4 bg-accent/10 border border-accent/20 rounded-lg">
          <div className="flex items-start gap-3">
            <Brain className="text-accent mt-1 flex-shrink-0" />
            <div className="text-sm">
              <p className="font-semibold mb-1">📄 Enhanced by Documentation</p>
              <p className="text-muted-foreground">
                3 additional services identified from your uploaded file
              </p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Right: Service Explanations */}
      <div className="space-y-4">
        {explanations.map((exp, i) => (
          <motion.div
            key={i}
            className="glass-card rounded-xl p-5"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.1, duration: 0.4 }}
          >
            <div className="flex items-start gap-3 mb-3">
              <div
                className={`${
                  categoryColors[exp.category]
                } w-10 h-10 rounded-lg flex items-center justify-center font-bold`}
              >
                {exp.icon}
              </div>
              <div className="flex-1">
                <h4 className="font-bold text-lg">{exp.service}</h4>
                <p className="text-sm text-muted-foreground">{exp.category}</p>
              </div>
            </div>

            <div className="mb-3">
              <p className="text-sm font-medium mb-2">
                We selected this because you mentioned:
              </p>
              <ul className="space-y-1">
                {exp.highlights.map((hl, j) => (
                  <li key={j} className="text-sm flex items-start gap-2">
                    <span className="text-accent">•</span>
                    <span
                      className={`${
                        categoryColors[exp.category]
                      } px-2 py-0.5 rounded`}
                    >
                      "{hl}"
                    </span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="mb-3">
              <p className="text-sm font-medium mb-2">Additional reasoning:</p>
              <ul className="space-y-1">
                {exp.reasons.map((reason, j) => (
                  <li key={j} className="text-sm flex items-start gap-2">
                    <span className="text-primary">→</span>
                    <span className="text-muted-foreground">{reason}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <div className="flex justify-between items-center mb-1 text-xs">
                <span className="text-muted-foreground">Confidence</span>
                <span className="font-semibold">{exp.confidence}%</span>
              </div>
              <Progress value={exp.confidence} className="h-1.5" />
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};
