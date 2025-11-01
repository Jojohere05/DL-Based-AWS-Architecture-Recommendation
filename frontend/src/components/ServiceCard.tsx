import { motion } from "framer-motion";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import { Progress } from "@/components/ui/progress";

interface ServiceCardProps {
  name: string;
  description: string;
  confidence: number;
  category: string;
  reasoning: string[];
  index: number;
}

const categoryColors: Record<string, string> = {
  Compute: "bg-blue-500",
  Storage: "bg-green-500",
  Database: "bg-orange-500",
  Networking: "bg-purple-500",
  Security: "bg-red-500",
  Analytics: "bg-cyan-500",
};

export const ServiceCard = ({
  name,
  description,
  confidence,
  category,
  reasoning,
  index,
}: ServiceCardProps) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <motion.div
      className="glass-card rounded-xl p-5 hover:shadow-elevated transition-all cursor-pointer"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1, duration: 0.4 }}
      whileHover={{ scale: 1.02 }}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Service Header */}
      <div className="flex items-start gap-4 mb-3">
        <div
          className={`${
            categoryColors[category] || "bg-gray-500"
          } w-12 h-12 rounded-lg flex items-center justify-center text-white font-bold text-lg`}
        >
          {name.substring(0, 2)}
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-bold mb-1">{name}</h3>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
      </div>

      {/* Confidence Score */}
      <div className="mb-3">
        <div className="flex justify-between items-center mb-2 text-sm">
          <span className="text-muted-foreground">Confidence</span>
          <span className="font-semibold">{confidence}%</span>
        </div>
        <Progress value={confidence} className="h-2" />
      </div>

      {/* Expandable Reasoning */}
      <div className="flex items-center justify-between text-sm text-primary font-medium">
        <span>Why selected?</span>
        {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </div>

      <motion.div
        initial={false}
        animate={{ height: expanded ? "auto" : 0, opacity: expanded ? 1 : 0 }}
        transition={{ duration: 0.3 }}
        className="overflow-hidden"
      >
        <ul className="mt-3 space-y-2 text-sm">
          {reasoning.map((reason, i) => (
            <li key={i} className="flex items-start gap-2">
              <span className="text-accent mt-1">•</span>
              <span className="text-muted-foreground">{reason}</span>
            </li>
          ))}
        </ul>
      </motion.div>
    </motion.div>
  );
};
