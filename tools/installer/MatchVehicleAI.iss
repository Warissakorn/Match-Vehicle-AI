; Inno Setup script for the thin (download-at-install) Windows installer.
;
; Compile with (ISCC.exe ships on GitHub's windows runners):
;   ISCC /DAppVersion=1.2.3 tools\installer\MatchVehicleAI.iss
;
; What this produces is around 20 MB against the packaged builds' 350 MB /
; 2.7 GiB -- and almost all of that 20 MB is uv.exe (49 MB on disk), not the
; application, which is about 1 MB of Python. The difference is not
; compression: the dependencies simply are not in here.
; Setup fetches them from PyPI and pytorch.org while it runs, picking the
; CPU or CUDA torch from the machine's own driver instead of asking the user
; to have chosen the right archive. See tools/installer/bootstrap.py for the
; trade-offs that buys and costs.

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif

#define AppName "Match-Vehicle-AI"
#define AppExeName "MatchVehicleAI.cmd"
; tools\installer\ -> repository root
#define RepoRoot "..\.."

[Setup]
AppId={{8F4C1E23-9B7A-4D62-A5E8-6C3F0B1D7A94}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Warissakorn
AppSupportURL=https://github.com/Warissakorn/Match-Vehicle-AI
DefaultDirName={autopf}\MatchVehicleAI
DefaultGroupName={#AppName}
OutputBaseFilename=MatchVehicleAI-windows-installer
OutputDir={#RepoRoot}
SetupIconFile={#RepoRoot}\assets\icon.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Per-user by default, so setup never raises a UAC prompt and the install
; needs no administrator. That fits what actually gets written: every
; user-writable thing the app owns (models, settings, logs, OCR caches)
; already lives under %LOCALAPPDATA% per user, so a machine-wide install
; would share only the read-only half. Users who do want it under Program
; Files can still say so -- the override below turns the choice into a
; dialog rather than removing it.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; The dependency download is the long pole here, not the disk write: a CUDA
; environment pulls roughly 2.5 GB. Say so before anything starts rather
; than after the user has committed to it.
DiskSpaceWarning=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; uv drives the whole provisioning step, so it has to be present before any
; of it runs -- it cannot itself be downloaded by the thing it bootstraps.
Source: "{#RepoRoot}\uv.exe"; DestDir: "{app}"; Flags: ignoreversion

; The application, which is the small part. Everything below is source that
; the environment built at install time runs directly.
Source: "{#RepoRoot}\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs
Source: "{#RepoRoot}\src\*"; DestDir: "{app}\src"; Flags: ignoreversion recursesubdirs
Source: "{#RepoRoot}\assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs
Source: "{#RepoRoot}\tools\install_torch.py"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "{#RepoRoot}\tools\pyinstaller_entry.py"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "{#RepoRoot}\tools\installer\bootstrap.py"; DestDir: "{app}\tools\installer"; Flags: ignoreversion
Source: "{#RepoRoot}\tools\installer\provision.cmd"; DestDir: "{app}\tools\installer"; Flags: ignoreversion
Source: "{#RepoRoot}\tools\installer\MatchVehicleAI.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\config.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\cli.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\extract_video.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\models_cli.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\requirements-lock.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; \
    IconFilename: "{app}\assets\icon.ico"; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; \
    IconFilename: "{app}\assets\icon.ico"; WorkingDir: "{app}"; Tasks: desktopicon

[UninstallDelete]
; Both directories are created *after* installation, by provision.cmd, so
; Inno has no record of their contents and a plain uninstall would leave
; gigabytes behind. They are listed explicitly for that reason. Nothing the
; user owns is in either -- models, settings and logs live under
; %LOCALAPPDATA%\MatchVehicleAI and are deliberately kept, so reinstalling
; does not re-download ~110 MB of model weights.
Type: filesandordirs; Name: "{app}\env"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\__pycache__"

[Code]
function ProvisionEnvironment(): Boolean;
var
  ResultCode: Integer;
begin
  { Not runhidden: this downloads and installs gigabytes and can run for
    many minutes. uv's and pip's own progress output in a visible console is
    the only signal the user gets that anything is happening -- a hidden
    window here is indistinguishable from a hung installer. }
  Result := Exec(ExpandConstant('{cmd}'),
                 ExpandConstant('/C ""{app}\tools\installer\provision.cmd" "{app}""'),
                 ExpandConstant('{app}'), SW_SHOW, ewWaitUntilTerminated, ResultCode);
  if Result then
    Result := (ResultCode = 0);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if not ProvisionEnvironment() then
      { Deliberately fatal rather than a warning that leaves a half-built
        install behind: without its environment the app cannot start at all,
        and an installed-looking Start Menu entry that fails on every click
        is worse than a setup that says it did not finish. }
      MsgBox('Setup could not download and install the dependencies.'#13#10#13#10
             'This step needs a working internet connection. The console '
             'window that just closed showed what failed.'#13#10#13#10
             'Match-Vehicle-AI is not usable until setup completes '
             'successfully -- please re-run the installer, or use the '
             'self-contained MatchVehicleAI-windows.zip from the Releases '
             'page instead.',
             mbCriticalError, MB_OK);
  end;
end;
