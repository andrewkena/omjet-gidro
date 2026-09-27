; Инсталлятор ОМДЖЕТ Гидро (Inno Setup 6). Собирается из уже готового
; dist/OMJET_Gidro.exe (см. README: pyinstaller OMJET_Gidro.spec) —
; сначала соберите exe, потом запустите:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" OMJET_Gidro.iss
; Готовый установщик появится в installer_dist/OMJET_Gidro_Setup_<версия>.exe

#define MyAppName "ОМДЖЕТ Гидро"
#define MyAppVersion "0.1.5"
#define MyAppPublisher "andrewkena"
#define MyAppURL "https://github.com/andrewkena/omjet-gidro"
#define MyAppExeName "OMJET_Gidro.exe"

[Setup]
AppId={{B6E7B7B2-6E3A-4C61-9C6C-6B8E9B7A0E1D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=installer_dist
OutputBaseFilename=OMJET_Gidro_Setup_{#MyAppVersion}
SetupIconFile=assets\gidro.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "Создать значок на рабочем столе"; GroupDescription: "Дополнительные значки:"

[Files]
Source: "dist\OMJET_Gidro.exe"; DestDir: "{app}"; Flags: ignoreversion
; coord_systems.txt хранится рядом с exe и правится пользователем руками —
; при обновлении установщик не должен затирать уже отредактированный файл.
Source: "coord_systems.txt"; DestDir: "{app}"; Flags: onlyifdoesntexist
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
