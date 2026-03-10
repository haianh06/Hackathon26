tải phần mềm rasberry pi imager v2.06 
vào cmd nhập: ssh pi@pi5.local
nếu thiếu key thì: ssh-keygen -R pi5.local sau đó lại ssh lại
tiếp theo nhập sudo raspi-config

sau khi vào được cmd của pi thì:
sudo apt update
sudo apt upgrade -y
sudo apt install code -y

tải lgpio và spidev
sudo apt install python3-lgpio -y
sudo apt install python3-spidev -y

git clone về và tạo môi trường:
python -m venv venv --system-site-packages

kích hoạt môi trường và tải các thư viện:
pip install numpy pandas streamlit

Chạy app:
streamlit run find_the_way/app.py