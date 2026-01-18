import random
import json
from datetime import datetime, timedelta
import uuid

# -----------------------------
# Root causes
# -----------------------------
ROOT_CAUSES = {
    "C1": "The serving cell's downtilt angle is too large, causing weak coverage at the far end.",
    "C2": "The serving cell's coverage distance exceeds 1km, resulting in over-shooting.",
    "C3": "A neighboring cell provides higher throughput.",
    "C4": "Non-colocated co-frequency neighboring cells cause severe overlapping coverage.",
    "C5": "Frequent handovers degrade performance.",
    "C6": "Neighbor cell and serving cell have the same PCI mod 30, leading to interference.",
    "C7": "Test vehicle speed exceeds 40km/h, impacting user throughput.",
    "C8": "Average scheduled RBs are below 160, affecting throughput."
}

PCIS = [44, 71, 113, 129, 258, 284, 432, 441, 712, 835]

# -----------------------------
# Helpers
# -----------------------------
def gen_timestamp(base, i):
    return (base + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S.000000")

def rand_coord(base, spread=0.001):
    return base + random.uniform(-spread, spread)

# -----------------------------
# Drive test generator
# -----------------------------
def gen_drive_test(cause, rows=10):
    base_time = datetime(2025, 5, 7, 13, 46, 2)
    serving_pci = random.choice(PCIS)
    neighbors = random.sample([p for p in PCIS if p != serving_pci], 4)

    lines = []
    for i in range(rows):
        speed = random.uniform(5, 35)
        rsrp = random.uniform(-82, -75)
        sinr = random.uniform(10, 22)
        rb = random.uniform(180, 220)
        tput = random.uniform(900, 1300)

        # RCA shaping
        if cause == "C1":
            rsrp -= i * 2
            tput -= i * 100

        elif cause == "C2":
            sinr = random.uniform(2, 8)
            tput = random.uniform(350, 600)

        elif cause == "C3":
            if i > rows // 2:
                serving_pci = neighbors[0]
                tput = random.uniform(900, 1300)

        elif cause == "C4":
            sinr = random.uniform(0, 4)

        elif cause == "C5":
            if i % 2 == 0:
                serving_pci = random.choice(neighbors)
            tput = random.uniform(300, 600)

        elif cause == "C6":
            sinr = random.uniform(1, 3)

        elif cause == "C7":
            speed = random.uniform(50, 90)
            tput = random.uniform(400, 1200)

        elif cause == "C8":
            rb = random.uniform(80, 140)
            tput = random.uniform(150, 500)

        line = (
            f"{gen_timestamp(base_time, i)}|"
            f"{rand_coord(128.1965):.6f}|{rand_coord(32.5828):.6f}|"
            f"{speed:.0f}|{serving_pci}|{rsrp:.2f}|{sinr:.2f}|{tput:.2f}|"
            f"{neighbors[0]}|{neighbors[1]}|{neighbors[2]}|{neighbors[3]}|-|"
            f"{rsrp-8:.2f}|{rsrp-15:.2f}|{rsrp-20:.2f}|{rsrp-25:.2f}|-|"
            f"{rb:.2f}"
        )
        lines.append(line)

    return "\n".join(lines)

# -----------------------------
# Engineering parameters
# -----------------------------
def gen_engineering(cause):
    mech_tilt = random.randint(3, 8)
    beam = "DEFAULT"

    if cause == "C1":
        mech_tilt = random.randint(12, 20)

    if cause == "C4":
        beam = "SCENARIO_8"

    return f"""gNodeB ID|Cell ID|Longitude|Latitude|Mechanical Azimuth|Mechanical Downtilt|Digital Tilt|Digital Azimuth|Beam Scenario|Height|PCI|TxRx Mode|Max Transmit Power|Antenna Model
0033916|27|128.197086|32.582192|330|{mech_tilt}|255|0|{beam}|14.0|{random.choice(PCIS)}|32T32R|34.9|NR AAU 3
"""

# -----------------------------
# Full user prompt
# -----------------------------
def build_prompt(cause):
    prompt = (
        "Analyze the 5G wireless network drive-test user plane data and engineering parameters.\n"
        "Identify the reason for the throughput dropping below 600Mbps in certain road sections.\n"
        "From the following 8 potential root causes, select the most likely one and enclose its number in \\boxed{{}} in the final answer.\n\n"
    )

    for k, v in ROOT_CAUSES.items():
        prompt += f"{k}: {v}\n"

    prompt += (
        "\nGiven:\n"
        "- The default electronic downtilt value is 255, representing a downtilt angle of 6 degrees. Other values represent the actual downtilt angle in degrees.\n\n"
        "Beam Scenario and Vertical Beamwidth Relationships:\n"
        "- When the cell's Beam Scenario is set to Default or SCENARIO_1 to SCENARIO_5, the vertical beamwidth is 6 degrees.\n"
        "- When the cell's Beam Scenario is set to SCENARIO_6 to SCENARIO_11, the vertical beamwidth is 12 degrees.\n"
        "- When the cell's Beam Scenario is set to SCENARIO_12 or above, the vertical beamwidth is 25 degrees.\n\n"
        "User plane drive test data as follows：\n\n"
        "Timestamp|Longitude|Latitude|GPS Speed (km/h)|5G KPI PCell RF Serving PCI|"
        "5G KPI PCell RF Serving SS-RSRP [dBm]|5G KPI PCell RF Serving SS-SINR [dB]|"
        "5G KPI PCell Layer2 MAC DL Throughput [Mbps]|"
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 PCI|"
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 2 PCI|"
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 PCI|"
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 4 PCI|"
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 5 PCI|"
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 Filtered Tx BRSRP [dBm]|"
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 2 Filtered Tx BRSRP [dBm]|"
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 Filtered Tx BRSRP [dBm]|"
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 4 Filtered Tx BRSRP [dBm]|"
        "Measurement PCell Neighbor Cell Top Set(Cell Level) Top 5 Filtered Tx BRSRP [dBm]|"
        "5G KPI PCell Layer1 DL RB Num (Including 0)\n"
    )

    prompt += gen_drive_test(cause)
    prompt += "\n\nEngeneering parameters data as follows：\n\n"
    prompt += gen_engineering(cause)

    return prompt

# -----------------------------
# Dataset writer (JSONL)
# -----------------------------
def generate_jsonl(samples_per_cause=200, outfile="synthetic_rca_chat.jsonl"):
    with open(outfile, "w", encoding="utf-8") as f:
        for cause in ROOT_CAUSES.keys():
            for _ in range(samples_per_cause):
                record = {
                    "messages": [
                        {
                            "role": "user",
                            "content": build_prompt(cause)
                        },
                        {
                            "role": "assistant",
                            "content": cause
                        }
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Generated {samples_per_cause * len(ROOT_CAUSES)} samples → {outfile}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    generate_jsonl(samples_per_cause=250)
