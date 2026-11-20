## 📌 Como usar

1. **Prepare um CSV** com colunas `image_path`, `identity` e `demographic_group`. Exemplo (FairFace):
   ```csv
   image_path,identity,demographic_group
   fotos/joao_1.jpg,joao,White
   fotos/joao_2.jpg,joao,White
   fotos/maria_1.jpg,maria,Black
   ...
   ```
2. **Instale as dependências**: `pip install -r fairness_audit/requirements_fairness.txt`.
3. **Execute cada teste**:
   ```bash
   python fairness_audit/audit_global.py --csv dados.csv --base-dir ./fotos
   python fairness_audit/audit_demographic.py --csv dados.csv --base-dir ./fotos
   python fairness_audit/audit_thresholds.py --csv dados.csv --base-dir ./fotos --target-far 1e-4
   # Para 1:N, forneça CSVs separados de galeria e queries
   python fairness_audit/audit_1toN.py --gallery-csv galeria.csv --query-csv queries.csv --base-dir ./fotos
   ```

Os resultados (CSV, PNG, TXT) serão salvos no prefixo indicado por `--output`. O teste 2 gera as disparidades no formato exigido pela ISO/IEC 19795-10 e o teste 3 evidencia se um threshold único é inadequado para a população brasileira.