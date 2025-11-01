import { Button } from "@/components/ui/button";
import { Save, RotateCcw, Share2, History } from "lucide-react";
import { toast } from "sonner";

export const Footer = () => {
  const handleSave = () => toast.success("Architecture saved!");
  const handleShare = () => toast.success("Share link copied to clipboard!");
  const handleNew = () => window.location.reload();

  return (
    <footer className="sticky bottom-0 z-40 glass-card border-t py-3">
      <div className="container mx-auto px-4">
        <div className="flex flex-wrap items-center justify-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={handleSave}
            className="flex items-center gap-2"
          >
            <Save size={16} />
            <span className="hidden sm:inline">Save Architecture</span>
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleNew}
            className="flex items-center gap-2"
          >
            <RotateCcw size={16} />
            <span className="hidden sm:inline">Start New</span>
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleShare}
            className="flex items-center gap-2"
          >
            <Share2 size={16} />
            <span className="hidden sm:inline">Share Link</span>
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="flex items-center gap-2"
          >
            <History size={16} />
            <span className="hidden sm:inline">History (3)</span>
          </Button>
        </div>
      </div>
    </footer>
  );
};
