/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#060B14',
          900: '#0B1120',
          850: '#0F172A',
          800: '#15203B',
          700: '#1E293B',
          600: '#334155',
          500: '#475569',
        },
        forensic: {
          real: '#10B981',       // Emerald Green
          fake: '#EF4444',       // Crimson Red
          edited: '#F97316',     // Bright Orange/Amber
          uncertain: '#EAB308',  // Yellow
          cyan: '#06B6D4',       // Technical Accent Cyan
          blue: '#3B82F6',       // Deep Blue
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'Courier New', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
