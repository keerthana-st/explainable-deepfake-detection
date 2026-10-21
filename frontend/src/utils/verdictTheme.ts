export type VerdictCategory = 'REAL' | 'AI-GENERATED' | 'AI-EDITED' | 'UNCERTAIN';

export interface VerdictTheme {
  category: VerdictCategory;
  label: string;
  badgeBg: string;
  badgeText: string;
  badgeBorder: string;
  glow: string;
  gradient: string;
  accentHex: string;
  lightBg: string;
  cardBorder: string;
}

export function getVerdictTheme(rawVerdict?: string): VerdictTheme {
  const v = (rawVerdict || '').toUpperCase().trim();

  if (v.includes('REAL') || v.includes('AUTHENTIC') || v.includes('NATURAL')) {
    return {
      category: 'REAL',
      label: 'REAL / AUTHENTIC',
      badgeBg: 'bg-emerald-500/10',
      badgeText: 'text-emerald-400',
      badgeBorder: 'border-emerald-500/30',
      glow: 'shadow-[0_0_25px_rgba(16,185,129,0.25)]',
      gradient: 'from-emerald-950/60 via-slate-900 to-slate-950',
      accentHex: '#10B981',
      lightBg: 'bg-emerald-950/20',
      cardBorder: 'border-emerald-500/40',
    };
  }

  if (v.includes('AI-EDITED') || v.includes('INPAINTED') || v.includes('MANIPULATED')) {
    return {
      category: 'AI-EDITED',
      label: 'AI-EDITED / INPAINTED',
      badgeBg: 'bg-orange-500/10',
      badgeText: 'text-orange-400',
      badgeBorder: 'border-orange-500/30',
      glow: 'shadow-[0_0_25px_rgba(249,115,22,0.25)]',
      gradient: 'from-orange-950/60 via-slate-900 to-slate-950',
      accentHex: '#F97316',
      lightBg: 'bg-orange-950/20',
      cardBorder: 'border-orange-500/40',
    };
  }

  if (v.includes('AI-GENERATED') || v.includes('FAKE') || v.includes('SYNTHETIC')) {
    return {
      category: 'AI-GENERATED',
      label: 'AI-GENERATED / SYNTHETIC',
      badgeBg: 'bg-rose-500/10',
      badgeText: 'text-rose-400',
      badgeBorder: 'border-rose-500/30',
      glow: 'shadow-[0_0_25px_rgba(239,68,68,0.25)]',
      gradient: 'from-rose-950/60 via-slate-900 to-slate-950',
      accentHex: '#EF4444',
      lightBg: 'bg-rose-950/20',
      cardBorder: 'border-rose-500/40',
    };
  }

  return {
    category: 'UNCERTAIN',
    label: 'UNCERTAIN / REVIEW REQUIRED',
    badgeBg: 'bg-amber-500/10',
    badgeText: 'text-amber-400',
    badgeBorder: 'border-amber-500/30',
    glow: 'shadow-[0_0_25px_rgba(234,179,8,0.25)]',
    gradient: 'from-amber-950/60 via-slate-900 to-slate-950',
    accentHex: '#EAB308',
    lightBg: 'bg-amber-950/20',
    cardBorder: 'border-amber-500/40',
  };
}
