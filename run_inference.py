import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./qwen_finetuned"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Load Model (in 4-bit to save memory)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Replace the output layer with a classification head
import torch.nn as nn
model.lm_head = nn.Linear(model.model.embed_tokens.embedding_dim, 8, bias=False).to(model.device, dtype=torch.bfloat16)

# 3. Prepare Input
messages = [
    {"role": "user", "content": "Analyze the 5G wireless network drive-test user plane data and engineering parameters.\nIdentify the reason for the throughput dropping below 600Mbps in certain road sections.\nFrom the following 8 potential root causes, select the most likely one and enclose its number in \\boxed{{}} in the final answer.\n\nC1: The serving cell's downtilt angle is too large, causing weak coverage at the far end.\nC2: The serving cell's coverage distance exceeds 1km, resulting in over-shooting.\nC3: A neighboring cell provides higher throughput.\nC4: Non-colocated co-frequency neighboring cells cause severe overlapping coverage.\nC5: Frequent handovers degrade performance.\nC6: Neighbor cell and serving cell have the same PCI mod 30, leading to interference.\nC7: Test vehicle speed exceeds 40km/h, impacting user throughput.\nC8: Average scheduled RBs are below 160, affecting throughput.\n\nGiven:\n- The default electronic downtilt value is 255, representing a downtilt angle of 6 degrees. Other values represent the actual downtilt angle in degrees.\n\nBeam Scenario and Vertical Beamwidth Relationships:\n- When the cell's Beam Scenario is set to Default or SCENARIO_1 to SCENARIO_5, the vertical beamwidth is 6 degrees.\n- When the cell's Beam Scenario is set to SCENARIO_6 to SCENARIO_11, the vertical beamwidth is 12 degrees.\n- When the cell's Beam Scenario is set to SCENARIO_12 or above, the vertical beamwidth is 25 degrees.\n\nUser plane drive test data as follows\uff1a\n\nTimestamp|Longitude|Latitude|GPS Speed (km/h)|5G KPI PCell RF Serving PCI|5G KPI PCell RF Serving SS-RSRP [dBm]|5G KPI PCell RF Serving SS-SINR [dB]|5G KPI PCell Layer2 MAC DL Throughput [Mbps]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 2 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 4 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 5 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 Filtered Tx BRSRP [dBm]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 2 Filtered Tx BRSRP [dBm]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 Filtered Tx BRSRP [dBm]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 4 Filtered Tx BRSRP [dBm]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 5 Filtered Tx BRSRP [dBm]|5G KPI PCell Layer1 DL RB Num (Including 0)\n2025-05-07 13:41:42.000000|128.197229|32.585111|31|441|-78.52|10.36|1136.62|226|13|480|-|-|-88.11|-96.63|-103.23|-|-|178.59\n2025-05-07 13:41:43.000000|128.198455|32.584408|29|441|-82.88|5.69|626.36|226|13|-|-|-|-82.35|-95.78|-|-|-|208.72\n2025-05-07 13:41:44.000000|128.197237|32.58508|21|441|-82.34|3.9|1143.77|226|13|793|-|-|-81.24|-96.17|-102.33|-|-|210.15\n2025-05-07 13:41:45.000000|128.197229|32.585069|32|441|-84.55|10.32|319.87|226|13|-|-|-|-79.82|-93.49|-|-|-|183.63\n2025-05-07 13:41:46.000000|128.197229|32.585038|30|226|-76.96|10.88|435.1|441|13|-|-|-|-87.35|-100.68|-|-|-|192.84\n2025-05-07 13:41:47.000000|128.197222|32.585019|3|441|-85.45|4.44|449.48|226|13|747|-|-|-91.41|-99.31|-104.23|-|-|206.35\n2025-05-07 13:41:48.000000|128.197222|32.585|26|226|-78.13|6.39|412.46|441|13|747|-|-|-86.45|-97.39|-103.7|-|-|191.74\n2025-05-07 13:41:49.000000|128.197207|32.584973|10|226|-75.37|11.83|1137.89|441|13|747|-|-|-88.84|-99.01|-103.8|-|-|198.58\n2025-05-07 13:41:50.000000|128.197192|32.58495|24|226|-69.9|16.38|1134.6|441|13|-|-|-|-84.99|-96.59|-|-|-|189.86\n2025-05-07 13:41:51.000000|128.197177|32.584912|8|226|-71.06|19.89|1097.91|441|13|747|-|-|-86.48|-99.45|-106.34|-|-|194.06\n\n\nEngeneering parameters data as follows\uff1a\n\ngNodeB ID|Cell ID|Longitude|Latitude|Mechanical Azimuth|Mechanical Downtilt|Digital Tilt|Digital Azimuth|Beam Scenario|Height|PCI|TxRx Mode|Max Transmit Power|Antenna Model\n0033918|38|128.195582|32.586069|180|20|6|0|SCENARIO_12|15.0|226|32T32R|34.9|NR AAU 1\n0033917|3|128.193759|32.576617|320|13|7|5|SCENARIO_2|31.5|264|64T64R|34.9|NR AAU 2\n0033913|25|128.208989|32.574724|10|16|13|10|SCENARIO_2|3.0|793|64T64R|34.9|NR AAU 2\n0033984|4|128.190867|32.589488|130|10|255|0|DEFAULT|78.0|480|64T64R|34.9|NR AAU 2\n0033918|13|128.197412|32.586605|180|2|0|0|SCENARIO_4|8.5|441|64T64R|34.9|NR AAU 2\n0033918|53|128.193256|32.586476|60|20|5|0|DEFAULT|38.0|688|32T32R|34.9|NR AAU 3\n0033980|38|128.198662|32.589556|295|12|6|0|SCENARIO_12|20.0|13|64T64R|34.9|NR AAU 2\n"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 4. Classify (using the new head)
print("\nRunning classification...")
with torch.no_grad():
    outputs = model(model_inputs.input_ids)
    # Get logits for the last token in the sequence
    # outputs.logits shape: [batch_size, seq_len, 8]
    last_token_logits = outputs.logits[:, -1, :]
    predicted_class = torch.argmax(last_token_logits, dim=-1).item()

print(model)
print(f"\nPredicted Class Index: {predicted_class}")
print(f"Logits: {last_token_logits.cpu()}")