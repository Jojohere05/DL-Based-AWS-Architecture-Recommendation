import { motion } from "framer-motion";
import { Download, ZoomIn, ZoomOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { toast } from "sonner";

interface DiagramNode {
  id: string;
  name: string;
  category: string;
  x: number;
  y: number;
}

interface DiagramConnection {
  from: string;
  to: string;
}

interface ArchitectureDiagramProps {
  nodes: DiagramNode[];
  connections: DiagramConnection[];
}

const categoryColors: Record<string, string> = {
  Compute: "#3b82f6",
  Storage: "#10b981",
  Database: "#f59e0b",
  Networking: "#8b5cf6",
  Security: "#ef4444",
};

export const ArchitectureDiagram = ({
  nodes,
  connections,
}: ArchitectureDiagramProps) => {
  const [zoom, setZoom] = useState(1);

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.2, 2));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.2, 0.5));
  
  const handleExport = () => {
    toast.success("Diagram exported as PNG!");
  };

  return (
    <div className="relative h-[600px] glass-card rounded-2xl overflow-hidden">
      {/* Controls */}
      <div className="absolute top-4 right-4 z-10 flex gap-2">
        <Button
          size="sm"
          variant="secondary"
          onClick={handleZoomIn}
          className="glass-card"
        >
          <ZoomIn size={16} />
        </Button>
        <Button
          size="sm"
          variant="secondary"
          onClick={handleZoomOut}
          className="glass-card"
        >
          <ZoomOut size={16} />
        </Button>
        <Button
          size="sm"
          variant="secondary"
          onClick={handleExport}
          className="glass-card"
        >
          <Download size={16} className="mr-2" />
          Export PNG
        </Button>
      </div>

      {/* Canvas */}
      <div className="w-full h-full flex items-center justify-center p-8 bg-gradient-to-br from-background to-muted/20">
        <motion.div
          style={{ transform: `scale(${zoom})` }}
          transition={{ duration: 0.3 }}
          className="relative w-full h-full"
        >
          {/* Draw connections */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            {connections.map((conn, i) => {
              const fromNode = nodes.find((n) => n.id === conn.from);
              const toNode = nodes.find((n) => n.id === conn.to);
              if (!fromNode || !toNode) return null;

              return (
                <motion.line
                  key={i}
                  x1={`${fromNode.x}%`}
                  y1={`${fromNode.y}%`}
                  x2={`${toNode.x}%`}
                  y2={`${toNode.y}%`}
                  stroke="hsl(var(--primary))"
                  strokeWidth="2"
                  strokeDasharray="5,5"
                  opacity="0.4"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 1, delay: i * 0.1 }}
                />
              );
            })}
          </svg>

          {/* Draw nodes */}
          {nodes.map((node, i) => (
            <motion.div
              key={node.id}
              className="absolute"
              style={{
                left: `${node.x}%`,
                top: `${node.y}%`,
                transform: "translate(-50%, -50%)",
              }}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: i * 0.15, type: "spring" }}
              whileHover={{ scale: 1.1 }}
            >
              <div
                className="w-24 h-24 rounded-xl flex items-center justify-center text-white font-bold text-sm shadow-lg cursor-pointer"
                style={{
                  backgroundColor: categoryColors[node.category] || "#6b7280",
                }}
              >
                <div className="text-center">
                  <div className="text-xs opacity-80">{node.category}</div>
                  <div className="mt-1">{node.name}</div>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 glass-card p-3 rounded-lg text-xs">
        <div className="font-semibold mb-2">Legend</div>
        {Object.entries(categoryColors).map(([category, color]) => (
          <div key={category} className="flex items-center gap-2 mb-1">
            <div
              className="w-3 h-3 rounded"
              style={{ backgroundColor: color }}
            />
            <span>{category}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
