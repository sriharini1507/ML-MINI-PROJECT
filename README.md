 Step 1: Install Ansible (Control Node)
sudo apt update
sudo apt install ansible -y
Check:
ansible --version
✅ Step 2: Enable SSH (Managed Node)
On server (managed node):

sudo apt install openssh-server -y
sudo systemctl start ssh
sudo systemctl enable ssh
✅ Step 3: Setup Passwordless SSH
On control node:

ssh-keygen
ssh-copy-id username@10.10.17.4
👉 Example:

ssh-copy-id Akshara@10.10.17.4
Test:

ssh Akshara@10.10.17.4
✅ Step 4: Create Inventory File
nano inventory.ini
Add:

[webservers]
10.10.17.4 ansible_user=Akshara
✅ Step 5: Test Connection
ansible all -i inventory.ini -m ping
👉 Output should show:

SUCCESS
✅ Step 6: Create Ansible Role
Run:

ansible-galaxy init my_web_role
👉 This creates folder structure:

my_web_role/
 ├── tasks/
 │    └── main.yml
 ├── handlers/
 ├── defaults/
 ├── vars/
 ├── meta/
✅ Step 7: Edit Role (Install Apache)
Open:

nano my_web_role/tasks/main.yml
Add:

---
- name: Install Apache
  apt:
    name: apache2
    state: present
    update_cache: yes

- name: Ensure Apache is running
  service:
    name: apache2
    state: started
    enabled: yes
✅ Step 8: Create Playbook
Create:

nano site.yml
Add:

---
- name: Configure web server
  hosts: webservers
  become: true

  roles:
    - my_web_role
✅ Step 9: Run Playbook
ansible-playbook -i inventory.ini site.yml -K
👉 -K → asks for sudo password

✅ Step 10: Verify Apache Installation
Login to server:

ssh Akshara@10.10.17.4
Check:

apache2 -v


