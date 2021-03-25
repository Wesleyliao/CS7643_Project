# Stops if error occurs
set -e

# Unzip the archives
for i in {1..13}
do
    s="00$i"
    unzip -o "./ffhq/drive-download-20210325T020201Z-${s: -3}.zip" -d ./ffhq/
done

# Move files into main directory
for i in {0..19}
do
    s="00${i}000"
    echo "Moving ./ffhq/${s: -5} contents to ./ffhq/"
    mv "./ffhq/${s: -5}"/* ./ffhq/
    rm -rf "./ffhq/${s: -5}"
done

# Delete archives
for i in {1..13}
do
    s="00$i"
    echo "Deleting ./ffhq/drive-download-20210325T020201Z-${s: -3}.zip"
    rm "./ffhq/drive-download-20210325T020201Z-${s: -3}.zip"
done