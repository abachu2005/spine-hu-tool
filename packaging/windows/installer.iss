; Inno Setup script for the Spine HU Tool Windows installer ("setup wizard").
;
; Build steps (from repo root, on Windows):
;   pip install pyinstaller
;   pyinstaller packaging\spine_hu.spec --noconfirm --distpath packaging\dist --workpath packaging\build
;   iscc packaging\windows\installer.iss
;
; Produces: packaging\dist\SpineHUTool-Setup.exe
; The PyInstaller onedir output lives in "packaging\dist\Spine HU Tool\".

#define AppName "Spine HU Tool"
; Kept in sync with spine_hu_tool.__version__ by packaging/sync_version.py
; (Inno Setup cannot read it directly). CI fails the build if this drifts.
#define AppVersion "0.1.3"
#define AppPublisher "Spine HU Tool"
#define AppExeName "Spine HU Tool.exe"

[Setup]
AppId={{B3D2A9F1-4C7E-4A2B-9E1D-7F0A6C3B2E55}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=SpineHUTool-Setup
SetupIconFile=..\icons\spine_hu.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#AppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Ship the entire PyInstaller onedir output.
Source: "..\dist\Spine HU Tool\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
