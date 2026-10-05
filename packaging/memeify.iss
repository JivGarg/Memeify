; Inno Setup script for memeify-setup.exe. Build with packaging\build.ps1 (needs dist\memeify from PyInstaller).
#define AppVersion "0.1.0"

[Setup]
AppId={{B3C1E8A4-5D2F-4C7B-9A61-2E4F0D7C8B15}
AppName=memeify
AppVersion={#AppVersion}
AppPublisher=Jivesh Garg
DefaultDirName={autopf}\memeify
DefaultGroupName=memeify
; Per-user install: no admin prompt, and the app can write its models, recordings and settings next to itself.
PrivilegesRequired=lowest
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=memeify-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=memeify
UninstallDisplayIcon={app}\memeify.exe
WizardStyle=modern

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\dist\memeify\*"; DestDir: "{app}"; Excludes: "models,models\*,recordings,recordings\*,memeify.log,.memeify-state.json,config.yaml"; Flags: recursesubdirs ignoreversion
; Keep the user's own settings when updating.
Source: "..\dist\memeify\config.yaml"; DestDir: "{app}"; Flags: onlyifdoesntexist

[Icons]
Name: "{group}\memeify"; Filename: "{app}\memeify.exe"
Name: "{group}\Uninstall memeify"; Filename: "{uninstallexe}"
Name: "{autodesktop}\memeify"; Filename: "{app}\memeify.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\memeify.exe"; Description: "Start memeify (it offers to install the memeify camera on first run)"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{app}\memeify.log"
Type: files; Name: "{app}\.memeify-state.json"
Type: filesandordirs; Name: "{app}\models"
; recordings\ is the user's own footage, so it stays.

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  code: Integer;
begin
  if (CurUninstallStep = usUninstall) and not UninstallSilent then
    if MsgBox('Also remove the "memeify" camera from Windows? (Windows will ask for permission.)',
              mbConfirmation, MB_YESNO) = IDYES then
      Exec(ExpandConstant('{app}\memeify.exe'), '--uninstall-camera', '', SW_HIDE, ewWaitUntilTerminated, code);
end;
