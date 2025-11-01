import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Cloud, Zap, TrendingUp } from "lucide-react";
import heroBackground from "@/assets/hero-background.jpg";

export const Hero = ({ onGetStarted }: { onGetStarted: () => void }) => {
  return (
    <section className="relative min-h-[80vh] flex items-center justify-center overflow-hidden">
      {/* Background Image with Overlay */}
      <div className="absolute inset-0 z-0">
        <img
          src={heroBackground}
          alt="Cloud Architecture Background"
          className="w-full h-full object-cover opacity-20"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-primary/20 via-background to-accent/10" />
      </div>

      {/* Floating Cloud Icons */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          className="absolute top-1/4 left-1/4 text-primary/10"
          animate={{ y: [0, -20, 0] }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
        >
          <Cloud size={80} />
        </motion.div>
        <motion.div
          className="absolute top-1/3 right-1/4 text-accent/10"
          animate={{ y: [0, 20, 0] }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        >
          <Cloud size={120} />
        </motion.div>
        <motion.div
          className="absolute bottom-1/4 right-1/3 text-primary/10"
          animate={{ y: [0, -15, 0] }}
          transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }}
        >
          <Cloud size={60} />
        </motion.div>
      </div>

      {/* Content */}
      <div className="container relative z-10 px-4 mx-auto text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            Transform Requirements into
            <span className="block gradient-text mt-2">
              Cloud Architecture in Seconds
            </span>
          </h1>

          <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-3xl mx-auto">
            AI-powered AWS service recommendations with visual diagrams - no
            cloud expertise required
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
            <Button
              size="lg"
              className="text-lg px-8 py-6 bg-gradient-to-r from-primary to-primary-glow hover:shadow-[0_0_40px_hsl(var(--primary-glow)/0.4)] transition-all duration-300 hover:scale-105"
              onClick={onGetStarted}
            >
              <Zap className="mr-2" />
              Generate Architecture
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="text-lg px-8 py-6 border-2 hover:bg-primary/5 hover:scale-105 transition-all"
              onClick={onGetStarted}
            >
              Upload Documentation
            </Button>
          </div>

          {/* Trust Indicators */}
          <motion.div
            className="flex flex-wrap justify-center gap-8 text-sm md:text-base"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.6 }}
          >
            <div className="flex items-center gap-2 glass-card px-4 py-2 rounded-full">
              <TrendingUp className="text-accent" size={20} />
              <span className="font-semibold">500+ Architectures Generated</span>
            </div>
            <div className="flex items-center gap-2 glass-card px-4 py-2 rounded-full">
              <Cloud className="text-accent" size={20} />
              <span className="font-semibold">92% Accuracy</span>
            </div>
            <div className="flex items-center gap-2 glass-card px-4 py-2 rounded-full">
              <Zap className="text-accent" size={20} />
              <span className="font-semibold">2s Response Time</span>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};
