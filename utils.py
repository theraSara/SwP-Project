import os
import glob
import re
import pandas as pd

def parse_lme_files(pattern="regression_output/*effects.txt"):
    """
    Parse R LME output files to extract:
    - AIC values
    - Model improvement over baseline (chi-square test)
    - Model improvement beyond cloze (chi-square test)
    - Fixed effect coefficients
    """
    aic_records = []
    improvement_records = []
    fixed_effect_records = []

    output_files = glob.glob(pattern)

    for filepath in output_files:
        filename = os.path.basename(filepath)

        model_base = (
            "GPT-2" if "gpt2" in filename.lower()
            else "Gemma-270M" if "270m" in filename.lower()
            else "Gemma-12B"
        )

        with open(filepath, "r") as f:
            content = f.read()
            lines = content.split("\n")

        current_mode = None

        for i, line in enumerate(lines):
            # Detect section headers
            stripped = line.strip()

            if "UNIDIRECTIONAL - 17k" in stripped:
                current_mode = "Uni (17k Word)"
            elif "BIDIRECTIONAL - 17k" in stripped:
                current_mode = "Bi (17k Word)"
            elif "UNIDIRECTIONAL - NLTK" in stripped:
                current_mode = "Uni (NLTK)"
            elif "BIDIRECTIONAL - NLTK" in stripped:
                current_mode = "Bi (NLTK)"
            elif re.match(r"^=+ UNIDIRECTIONAL =+$", stripped):
                current_mode = "Uni (Full)"
            elif re.match(r"^=+ BIDIRECTIONAL =+$", stripped):
                current_mode = "Bi (Full)"

            if not current_mode:
                continue

            # Extract AIC from model summary
            if "AIC       BIC" in line:
                try:
                    numbers = lines[i + 1].split()
                    aic_records.append({
                        "Model_Base": model_base,
                        "Condition": current_mode,
                        "AIC": float(numbers[0]),
                        "BIC": float(numbers[1]),
                        "logLik": float(numbers[2]),
                    })
                except (IndexError, ValueError):
                    pass

            # Extract improvement over baseline
            if "improve over baseline" in line:
                # Look for the chi-square comparison a few lines down
                for j in range(i + 1, min(i + 15, len(lines))):
                    if "Chisq" in lines[j] and "Pr(>Chisq)" in lines[j]:
                        # Next lines have the model comparison
                        for k in range(j + 1, min(j + 4, len(lines))):
                            if "***" in lines[k] or "**" in lines[k] or "*" in lines[k] or "." in lines[k]:
                                parts = lines[k].split()
                                try:
                                    chisq_idx = next(
                                        idx for idx, p in enumerate(parts)
                                        if re.match(r"^\d+\.\d+$", p) and float(p) > 0
                                    )
                                    improvement_records.append({
                                        "Model_Base": model_base,
                                        "Condition": current_mode,
                                        "Test": "vs Baseline",
                                        "Chisq": float(parts[chisq_idx]),
                                        "p_value": parts[-1] if "e-" in parts[-1] or parts[-1] == "***" else parts[-2],
                                        "Significant": "***" in lines[k] or "**" in lines[k],
                                    })
                                except (StopIteration, ValueError, IndexError):
                                    pass
                        break

            # Extract improvement beyond cloze
            if "not captured by humans" in line:
                for j in range(i + 1, min(i + 15, len(lines))):
                    if "Chisq" in lines[j] and "Pr(>Chisq)" in lines[j]:
                        for k in range(j + 1, min(j + 4, len(lines))):
                            parts = lines[k].split()
                            if len(parts) >= 5:
                                try:
                                    # Find the Chisq value
                                    for idx, p in enumerate(parts):
                                        if re.match(r"^\d+\.\d+$", p):
                                            chisq_val = float(p)
                                            sig = "*" in lines[k].rstrip()
                                            improvement_records.append({
                                                "Model_Base": model_base,
                                                "Condition": current_mode,
                                                "Test": "Beyond Cloze",
                                                "Chisq": chisq_val,
                                                "p_value": parts[-1].rstrip(),
                                                "Significant": "***" in lines[k] or "**" in lines[k] or (lines[k].rstrip().endswith("*") and not lines[k].rstrip().endswith("**")),
                                            })
                                            break
                                except (ValueError, IndexError):
                                    pass
                        break

            # Extract fixed effects for surprisal predictor
            for prefix in ["s_g12b_uni", "s_g12b_bi", "s_g270_uni", "s_g270_bi",
                           "s_gpt2_uni", "s_gpt2_bi"]:
                if line.strip().startswith(prefix):
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            fixed_effect_records.append({
                                "Model_Base": model_base,
                                "Condition": current_mode,
                                "Predictor": prefix,
                                "Estimate": float(parts[1]),
                                "Std_Error": float(parts[2]),
                                "t_value": float(parts[4]),
                                "Significant": "***" in line or "**" in line or line.rstrip().endswith("*"),
                            })
                        except (ValueError, IndexError):
                            pass

    return (
        pd.DataFrame(aic_records),
        pd.DataFrame(improvement_records),
        pd.DataFrame(fixed_effect_records),
    )